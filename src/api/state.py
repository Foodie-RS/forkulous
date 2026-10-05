from __future__ import annotations

import abc
from collections.abc import Generator
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

    def move_from(self, child_state:RequestState, parent_state:RequestState, overwrite:bool=False):
        if child_state not in self._state_map:
            return
        if parent_state in self._state_map and not overwrite:
            if (self._state_map[child_state] != self._state_map[parent_state]):
                raise ValueError("Child state is not allowed to overwrite parent state!")
            return
        self._state_map[parent_state] = self._state_map[child_state]

    def get(self, state: RequestState, allow_deferred:bool, owner_state:RequestState|None, children:list[RequestState]) -> tuple[Option[T], _PromiseState]:
        if len(set(self._state_map.keys()).intersection(children)) > 0:
            return Option.none(), _PromiseState.CHILD_PROVIDES
        if owner_state is None and state in self._state_map:
            return self._state_map[state], _PromiseState.AVAIL
        elif owner_state is not None and owner_state in self._state_map:
            return self._state_map[owner_state], _PromiseState.AVAIL
        assert self._provider is not None
        if self._provider.deferred() and not allow_deferred:
            return Option.none(), _PromiseState.DEFER
        if isinstance(self._provider, OptionalProvider):
            res = cast(Option[T], self._provider(state))
        else:
            res = Option.some(self._provider(state))
        self._state_map[state] = res
        return res, _PromiseState.RESOLVED

    def provider(self) -> Provider[T] | OptionalProvider[T] | None:
        return self._provider

    def copy_for(self, old_state:RequestState, new_state:RequestState) -> _StorePromise[T]:
        state_map:weakref.WeakKeyDictionary[RequestState, Option[T]] = weakref.WeakKeyDictionary()
        if old_state in self._state_map:
            state_map[new_state] = self._state_map[old_state]
        return _StorePromise(
            _provider = self._provider,
            _state_map = state_map
        )

class _PromiseState(Enum):
    # Returned by get() if a matching value was found in the cache.
    AVAIL=1,
    # Returned by get() if a value has been obtained by calling the provider.
    RESOLVED=2,
    DEFER=3,
    # Returned by get() if a mask (child) RequestState already has a value available for this Promise.
    # This is to avoid duplicates.
    CHILD_PROVIDES=4

class RequestState:
    _masker: weakref.ReferenceType[RequestState] | None
    _masking: RequestState | None = None
    _store: dict[type[Any], list[_StorePromise[Any]]]
    _lock: RLock
    _poisoned: bool = False

    def __init__(self) -> None:
        self._store = {}
        self._masker = None
        self._lock = RLock()

    def set[T](self, key: type[T], item: T, append:bool=False):
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("set")
        with self._lock:
            self.assert_valid()
            self.assert_not_masked()
            origin = get_origin(key) or key
            if not isinstance(item, origin):
                raise TypeError("Item is not instance of key class")

            prom:_StorePromise[T] = _StorePromise()
            prom.set(self, Option.some(item))
            lst:list[_StorePromise[T]] = cast(list[_StorePromise[T]], self._store.setdefault(key, []))
            if append:
                logger.log(5, f"Appending value for {key} (origin: {origin})")
                lst.append(prom)
            else:
                logger.log(5, f"Inserting value for {key} (origin: {origin})")
                lst.insert(0, prom)

    def add_provider[T](
        self,
        key: type[T],
        callback: Provider[T] | OptionalProvider[T],
        append:bool=True
    ):
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("set_prov")
        with self._lock:
            self.assert_valid()
            self.assert_not_masked()
            origin = get_origin(key) or key
            if not callback.provides(origin):
                raise TypeError("Provider does not provide specified key")
            prom:_StorePromise[T] = _StorePromise(_provider=callback)

            lst:list[_StorePromise[T]] = cast(list[_StorePromise[T]], self._store.setdefault(key, []))
            if append:
                logger.log(5, f"Appending provider for {key} (origin: {origin})")
                lst.append(prom)
            else:
                logger.log(5, f"Inserting provider for {key} (origin: {origin})")
                lst.insert(0, prom)

    def _iter[T](self, what:type[T], resolve_deferred:bool, caller:RequestState, children:list[RequestState]) -> Generator[tuple[Option[T], _PromiseState], None, None]:
        logger = logging.getLogger("ingr_api").getChild("ReqState").getChild("iter")
        with self._lock:
            if what in self._store:
                lst:list[_StorePromise[T]] = cast(list[_StorePromise[T]], self._store[what])
                logger.log(5, f"Found {len(lst)} promises, yielding (state depth: {len(children)})...")
                for prom in lst:
                    res = prom.get(self if caller == self else caller, resolve_deferred, None if caller==self else self, children)
                    yield res
            if self._masking is not None:
                logger.log(5, "Calling parent state")
                yield from self._masking._iter(what, resolve_deferred, caller, children + [self])

    def iter[T](self, what:type[T], resolve_deferred:bool=True) -> Generator[T, None, None]:
        with self._lock:
            self.assert_valid()
            for res, _ in self._iter(what, resolve_deferred, self, []):
                if res.is_some():
                    yield(res.unwrap())

    def get_optional[T](self, what: type[T], resolve_deferred:bool=True) -> Option[T]:
        with self._lock:
            nxt = next(self.iter(what, resolve_deferred), None)
            return Option.some(nxt) if nxt is not None else Option.none()

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

    def commit(self, append:bool=True, overwrite:bool=False):
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
            for k,lst in self._store.items():
                parent_lst = self._masking._store.setdefault(k, [])
                for prom in lst:
                    prom.move_from(self, self._masking, overwrite)
                    if prom not in parent_lst:
                        if append:
                            parent_lst.append(prom)
                        else:
                            parent_lst.insert(0, prom)
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
            nxt = next(self.iter(what, allow_deferred), None)
            if nxt is None:
                raise ValueError(f"State contains no values of type {what}!")
            return nxt

    def get_all[T](self, what: type[T], allow_deferred:bool=False) -> list[T]:
        with self._lock:
            self.assert_valid()
            return [k for k in self.iter(what, allow_deferred)]

    def copy(self) -> RequestState:
        new_state = RequestState()
        new_store:dict[type[Any], list[_StorePromise[Any]]] = {}
        for k,v in self._store.items():
            new_lst:list[_StorePromise[Any]] = [it.copy_for(self, new_state) for it in v]
            new_store[k] = new_lst
        new_state._store = new_store
        return new_state

    def __contains__(self, what: type[Any]):
        with self._lock:
            self.assert_valid()
            return what in self._store or (
                self._masking is not None and what in self._masking
            )
