from __future__ import annotations

import abc
import logging
import weakref
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, cast, get_origin, override

from api.common import Option


@dataclass
class Provider[T](metaclass=abc.ABCMeta):
    """
    A Provider is an object that can *provide* something.
    Providers are the backbone of Forkulous: everything is
    produced by a Provider.

    To create a new Provider, simply create a class that
    inherits from Provider, set the _type and override the
    .execute() method. Then register the new Provider on the
    in the main API.

    It's very important that Providers respect typing.
    In particular, a Provider should never return None.
    If you need to return a None value, use OptionalProvider
    and Option.none() instead.
    """
    _type: type[T]

    def deferred(self) -> bool:
        """
        This method should return True if it depends on the result
        of other Providers of the same type.
        """
        return False

    def provides(self, what: type) -> bool:
        return what == self._type

    @abc.abstractmethod
    def execute(self, state: RequestState) -> T:
        """
        This method should always return a value (or raise an error).
        It's very important that typing is respected. This method should
        never return an object that is not an instance or a subclass of T.

        If you need to return a None value, use OptionalProvider and
        Option.none() instead.
        """

    def __call__(self, state: RequestState) -> T:
        return self.execute(state)


class OptionalProvider[T](Provider[Option[T]], metaclass=abc.ABCMeta):
    """
    A version of Provider that can return a wrapped
    None value, using Option.none(). If you return a
    not-None value, use Option.some(value) to wrap it.
    """
    @override
    def provides(self, what:type) -> bool:
        return Option[what] == self._type


@dataclass(frozen=True)
class _Null:
    pass


# Some providers may return None. To differentiate between a "None"
# that was returned from a correctly called provider, and the state of
# not having been called, we need a second null value.
_NULL = _Null()

@dataclass
class _StorePromise[T]:
    """
    A wrapper object for a Provider that may or may not have
    been called. Use .get() to either get the cached result,
    or to execute the provider and cache the result.
    """
    _provider: Provider[T] | OptionalProvider[T] | None = None
    _state_map:weakref.WeakKeyDictionary[RequestState, Option[T]] = field(default_factory=weakref.WeakKeyDictionary)

    def set(self, state:RequestState, val:Option[T]):
        self._state_map[state] = val

    def is_available(self, state:RequestState) -> bool:
        return state in self._state_map

    def get(self, state: RequestState, allow_deferred:bool, owner_state:RequestState|None, children:list[RequestState]) -> Option[T]|_Null:
        if len(set(self._state_map.keys()).intersection(children)) > 0:
            return _NULL
        if owner_state is None and state in self._state_map:
            return self._state_map[state]
        elif owner_state is not None and owner_state in self._state_map:
            return self._state_map[owner_state]
        assert self._provider is not None
        if self._provider.deferred() and not allow_deferred:
            return _NULL
        if isinstance(self._provider, OptionalProvider):
            res = cast(Option[T], self._provider(state))
        else:
            res = Option.some(self._provider(state))
        self._state_map[state] = res
        return res

    def provider(self) -> Provider[T] | OptionalProvider[T] | None:
        return self._provider

    def copy(self) -> _StorePromise[T]:
        state_map:weakref.WeakKeyDictionary[RequestState, Option[T]] = weakref.WeakKeyDictionary()
        state_map.update(self._state_map)
        return _StorePromise(
            _provider = self._provider,
            _state_map = state_map
        )

class _PromiseState(Enum):
    AVAIL=1,
    DEFER=2,
    NOT_FOUND=3

class RequestState:
    _masker: weakref.ReferenceType[RequestState] | None
    _masking: RequestState | None = None
    _store: dict[type[Any], _StorePromise[Any]]
    _multi_store: dict[type[Any], list[_StorePromise[Any]]]
    _lock: RLock
    _poisoned: bool = False

    def __init__(self) -> None:
        self._store = {}
        self._multi_store = {}
        self._masker = None
        self._lock = RLock()

    def set[T](self, key: type[T], item: T):
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("set")
        with self._lock:
            self.assert_valid()
            self.assert_not_masked()
            origin = get_origin(key) or key
            if not isinstance(item, origin):
                raise TypeError("Item is not instance of key class")
            if key in self._store:
                it = self._store.pop(key)
                logger.log(5, f"Cleared: {key} (available: {it.is_available(self)})")
            self._store[key] = _StorePromise()
            self._store[key].set(self, Option.some(item))

    def set_provider[T](
        self,
        key: type[T],
        callback: Provider[T] | OptionalProvider[T],
        overwrite_available: bool = False,
    ):
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("set_prov")
        with self._lock:
            self.assert_valid()
            self.assert_not_masked()
            origin = get_origin(key) or key
            if not callback.provides(origin):
                raise TypeError("Provider does not provide specified key")
            if key in self._store:
                it = self._store[key]
                if it.is_available(self) and not overwrite_available:
                    logger.log(
                        5, f"Not storing callback for {key}, value already available"
                    )
                    return
            self._store[key] = _StorePromise(_provider=callback)

    def _get_masked[T](
        self, what: type[T], orig_state: RequestState, allow_deferred:bool, children: list[RequestState]
    ) -> tuple[Option[T], _PromiseState]:
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("get_mskd")
        with self._lock:
            self.assert_valid()
            if what in self._store:
                promise: _StorePromise[T] = self._store[what]
                if not promise.is_available(self):
                    logger.log(5, f"Calling provider: {what}")
                else:
                    logger.log(5, f"Promise available: {what}")
                value = promise.get(orig_state, allow_deferred, self, children)
                if isinstance(value, _Null):
                    return Option.none(), _PromiseState.DEFER
                return value, _PromiseState.AVAIL
            if self._masking is not None:
                logger.log(5, f"Recursing into masked state for {what}")
                return self._masking._get_masked(what, orig_state, allow_deferred, children +  [self])
            return Option.none(), _PromiseState.NOT_FOUND

    def _multi_get_masked[T](
        self, what: type[T], orig_state: RequestState, resolve_deferred:bool, children:list[RequestState]
    ) -> list[tuple[Option[T], _PromiseState]]:
        with self._lock:
            self.assert_valid()
            logger = (
                logging.getLogger("ingr_api")
                .getChild("ReqState")
                .getChild("mlt_get_mskd")
            )
            res: list[tuple[Option[T], _PromiseState]] = []
            proms = self._multi_store.setdefault(what, [])
            for prom in proms:
                val: Option[T]|_Null = prom.get(orig_state, resolve_deferred, self, children)
                if isinstance(val, _Null):
                    res.append((Option.none(), _PromiseState.DEFER))
                else:
                    res.append((val, _PromiseState.AVAIL))
            if self._masking is not None:
                logger.log(5, f"Recursing into masked state for {what}")
                res2 = self._masking._multi_get_masked(what, orig_state, resolve_deferred, children + [self])
                res.extend(res2)
            return res

    def _get[T](self, what:type[T], resolve_deferred:bool) -> tuple[Option[T], _PromiseState]:
        with self._lock:
            self.assert_valid()
            logger = (
                logging.getLogger("ingr_api").getChild("ReqState").getChild("get_opt")
            )
            if what in self._store:
                promise: _StorePromise[T] = self._store[what]
                if not promise.is_available(self):
                    logger.log(5, f"Calling provider: {what}")
                else:
                    logger.log(5, f"Promise available: {what}")
                value = promise.get(self, resolve_deferred, None, [])
                if isinstance(value, _Null):
                    logger.log(5, "Promise is deferred")
                    return Option.none(), _PromiseState.DEFER
                return value, _PromiseState.AVAIL
            logger.log(5, f"Store miss: {what}")
            if self._masking is not None:
                logger.log(5, f"Calling masked state for {what}")
                res, state = self._masking._get_masked(what, self, resolve_deferred, [self])
                if state == _PromiseState.DEFER:
                    logger.log(5, "Masked deferred hit")
                elif isinstance(res, _Null): # instance check instead of state check so pyright is happy
                    logger.log(5, "Masked miss")
                    return Option.none(), _PromiseState.NOT_FOUND
                else:
                    logger.log(5, "Masked hit")
                    self._store[what] = _StorePromise()
                    self._store[what].set(self, res)
                    return res, _PromiseState.AVAIL
            return Option.none(), _PromiseState.NOT_FOUND

    def get_optional[T](self, what: type[T], resolve_deferred:bool=False) -> Option[T]:
        res, _ = self._get(what, resolve_deferred)
        return res

    def invalidate(self):
        with self._lock:
            self._poisoned = True
            if self._masking is not None:
                super_msk = None if self._masking._masker is None else self._masking._masker()
                if super_msk != self:
                    raise ValueError("Invalid state: masker has been modified")
                else:
                    self._masking._masker = None
                    self._masking = None

    def is_poisoned(self) -> bool:
        poison: bool = self._poisoned or (
            self._masking is not None and self._masking.is_poisoned()
        )
        if poison:
            self._poisoned = True
        return poison

    def is_masked(self) -> bool:
        with self._lock:
            msk = None if self._masker is None else self._masker()
            return msk is not None and not msk.is_poisoned()

    def assert_valid(self):
        if self.is_poisoned():
            raise ValueError("State is poisoned")

    def assert_not_masked(self):
        if self.is_masked():
            raise ValueError("State is being masked and cannot be modified.")

    def commit(self):
        """
        Merges this state into the masked state.
        See #.mask() for an explanation of masking.
        This method will poison the state,
        which means that it cannot be used anymore and
        calls to any methods except #.poison() and
        #.is_invalid() will raise a ValueError.
        It is not possible nor intended to
        de-poison a RequestState, because it can
        lead to multiple divergent states.
        """
        with self._lock:
            self.assert_valid()
            if self.is_masked():
                raise ValueError(
                    "Called commit() on a masked RequestState. Needs to be called on the masking state."
                )
            if self._masking is None:
                raise ValueError("Called commit() on a non-masking RequestState")
            self._masking._store = self._masking._store | self._store
            for k in self._multi_store:
                lst = self._masking._multi_store.setdefault(k, [])
                lst.extend(self._multi_store[k])
            self.invalidate()

    def mask(self, invalidate_previous: bool = True) -> RequestState:
        """
        Create a new RequestState ("mask") that propagates to this RequestState.
        Modifications to the new RequestState will not affect the old ("masked") state,
        but calls to #get()-methods will propagate to it.
        Promises will be evaluated with - and their results stored in - the mask,
        even if they were originally stored in the masked RequestState.
        This means that promises that are evaluated on the mask will
        need to be reevaluated if the mask is not dropped or invalidated.
        While the mask is not invalidated, the masked RequestState cannot be modified
        (except to fulfill a promise).

        Call #commit() to write all contents of the mask to the original RequestState,
        or #invalidate() to roll back.
        """
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("mask")
        logger.log(5, "Mask called")
        with self._lock:
            self.assert_valid()
            if not invalidate_previous:
                self.assert_not_masked()
            elif self._masker is not None:
                msk = self._masker()
                if msk is not None:
                    msk.invalidate()
            new_state = RequestState()
            new_state._masking = self
            self._masker = weakref.ref(new_state)
            return new_state

    def get[T](self, what: type[T], allow_deferred:bool=False) -> T:
        with self._lock:
            self.assert_valid()
            if what not in self:
                raise IndexError("Class object is not in state dict")
            res:Option[T]
            state:_PromiseState
            (res, state) = self._get(what, resolve_deferred=allow_deferred)
            if res.is_none():
                if state == _PromiseState.DEFER:
                    raise ValueError("Promise was deferred, but allow_deferred was False!")
                else:
                    raise ValueError("Provider returned a None value!")
            return res.unwrap()

    def multi_get_opt[T](
        self, what: type[T], skip_none: bool = True, allow_deferred:bool=False
    ) -> list[Option[T]]:
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("m_get_o")
        with self._lock:
            self.assert_valid()
            res: list[Option[T]] = []
            if what in self._multi_store:
                promises: list[_StorePromise[T]] = self._multi_store[what]
                logger.log(5, f"Evaluating {len(promises)} promises: {what}")
                for prom in promises:
                    it = prom.get(self, allow_deferred, None, [])
                    if isinstance(it, _Null):
                        logger.log(5, "Promise reported deferred, skipping")
                    elif skip_none and it.is_none():
                        logger.log(5, "Got None, skipping")
                    else:
                        res.append(it)
            else:
                logger.log(5, f"No promises stored for {what}")
            if self._masking is not None:
                logger.log(5, f"Calling masked state for {what}")
                masked_res = self._masking._multi_get_masked(what, self, allow_deferred, [self])
                if skip_none:
                    masked_res = [k for k in masked_res if k[0].is_some()]
                logger.log(5, f"Got {len(masked_res)} results from masked")
                res.extend([k[0] for k in masked_res])
            return res

    def multi_get[T](self, what: type[T], skip_none: bool = True, allow_zero_len:bool=False, allow_deferred:bool=False) -> list[T]:
        with self._lock:
            o = self.multi_get_opt(what, skip_none=skip_none, allow_deferred=allow_deferred)
            res = [k.unwrap() for k in o if k.is_some()]
            if len(res) == 0 and not allow_zero_len:
                raise ValueError("No results")
            return res

    def _masked_multi_get_first[T](
        self, what: type[T], orig_state: RequestState, allow_deferred:bool, children:list[RequestState]
    ) -> Option[T]:
        logger = (
            logging.getLogger("ingr_api").getChild("ReqState").getChild("msk_m_get_f")
        )
        key = what
        with self._lock:
            self.assert_valid()
            if key in self._multi_store:
                logger.log(5, f"Hit: {key}")
                promises: list[_StorePromise[T]] = self._multi_store[key]
                for prom in promises:
                    res = prom.get(orig_state, allow_deferred, self, children)
                    if not isinstance(res, _Null) and res.is_some():
                        logger.log(5, "Not none, success")
                        return res
            if self._masking is not None:
                return self._masking._masked_multi_get_first(what, orig_state, allow_deferred, children + [self])
            return Option.none()

    def multi_get_first_opt[T](self, what: type[T], resolve_deferred:bool=False) -> Option[T]:
        # TODO: merge with get()
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("m_get_f")
        with self._lock:
            self.assert_valid()
            if what in self._multi_store:
                logger.log(5, f"Hit: {what}")
                promises: list[_StorePromise[T]] = self._multi_store[what]
                for prom in promises:
                    res = prom.get(self, resolve_deferred, None, [])
                    if (not isinstance(res, _Null)) and res.is_some():
                        logger.log(5, "Not none, success")
                        return res
                logger.log(5, f"No non-None values for {what}")
            else:
                logger.log(5, f"Miss: {what}")
            if self._masking is not None:
                logger.log(5, "Calling parent state")
                return self._masking._masked_multi_get_first(what, self, allow_deferred=resolve_deferred, children=[self])

            return Option.none()

    def multi_get_first[T](self, what: type[T], resolve_deferred:bool=False) -> T:
        it = self.multi_get_first_opt(what, resolve_deferred)
        res = it.expect("No provider returned a value")
        return res

    def multi_add[T](self, key: type[T], val: T, insert_first: bool = False):
        with self._lock:
            logger = (
                logging.getLogger("ingr_api").getChild("ReqState").getChild("m_add")
            )
            self.assert_valid()
            self.assert_not_masked()
            origin = get_origin(key) or key
            if not isinstance(val, origin):
                raise TypeError("Value must be instance of key")
            lst = self._multi_store.setdefault(key, [])
            prom:_StorePromise[T] = _StorePromise()
            prom.set(self, Option.some(val))
            if insert_first:
                logger.log(5, f"Inserting for {key}")
                lst.insert(0, prom)
            else:
                logger.log(5, f"Appending for {key}")
                lst.append(prom)

    def multi_add_prov[T](
        self,
        key: type[T],
        val: Provider[T] | OptionalProvider[T],
        insert_before: Provider[T] | OptionalProvider[T] | None = None,
        insert_after: Provider[T] | OptionalProvider[T] | None = None,
        insert_first: bool = False,
        fail_if_insert_not_found: bool = True,
    ):
        logger = (
            logging.getLogger("ingr_api").getChild("ReqState").getChild("m_add_prov")
        )
        with self._lock:
            self.assert_valid()
            self.assert_not_masked()
            if (
                (insert_after is not None and insert_first)
                or (insert_before is not None and insert_first)
                or (insert_before is not None and insert_after is not None)
            ):
                raise ValueError(
                    "Can only specify one of insert_first, insert_before or insert_after"
                )
            lst: list[_StorePromise[T]] = cast(
                list[_StorePromise[T]], self._multi_store.setdefault(key, [])
            )
            new_promise = _StorePromise(_provider=val)
            if insert_after is not None or insert_before is not None:
                for ix, it in enumerate(lst):
                    if (insert_before is not None) and (it.provider() == insert_before):
                        logger.log(5, f"Inserting for {key} before specified provider")
                        lst.insert(ix, new_promise)
                        return
                    elif (insert_after is not None) and (it.provider() == insert_after):
                        logger.log(5, f"Inserting for {key} after specified provider")
                        lst.insert(ix + 1, new_promise)
                        return
                if fail_if_insert_not_found:
                    raise ValueError(
                        f"Tried to insert provider for {key}, but insertion point was not found"
                    )
                else:
                    logger.log(
                        5,
                        f"Appending provider for {key} because insertion point was not found",
                    )
                    lst.append(new_promise)
            elif insert_first:
                logger.log(5, f"Inserting provider for {key} at first position")
                lst.insert(0, new_promise)
            else:
                logger.log(5, f"Appending provider for {key}")
                lst.append(new_promise)

    def copy(self) -> RequestState:
        new_state = RequestState()
        new_store:dict[type[Any], _StorePromise[Any]] = {}
        new_store.update([(k, v.copy()) for k, v in self._store.items()])
        new_multi_store:dict[type[Any], list[_StorePromise[Any]]] = {}
        for k,v in self._multi_store.items():
            new_lst:list[_StorePromise[Any]] = []
            new_lst.extend([it.copy() for it in v])
            new_multi_store[k] = new_lst
        new_state._store = new_store
        new_state._multi_store = new_multi_store
        return new_state

    def __contains__(self, what: type[Any]):
        with self._lock:
            self.assert_valid()
            return what in self._store or (
                self._masking is not None and what in self._masking
            )
