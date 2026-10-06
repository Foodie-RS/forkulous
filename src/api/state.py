from __future__ import annotations

import abc
import logging
import weakref
from collections.abc import Generator
from dataclasses import dataclass, field
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

class GeneratorProvider[T](metaclass=abc.ABCMeta):
    _type: type[T]

    def provides(self, what: type) -> bool:
        return what == self._type

    @abc.abstractmethod
    def execute(self, state:RequestState) -> Generator[T, None, None]:
        pass

class OptionalProvider[T](Provider[Option[T]], metaclass=abc.ABCMeta):
    """
    A version of Provider that can return a wrapped
    None value, using Option.none(). If you return a
    not-None value, use Option.some(value) to wrap it.
    """
    @override
    def provides(self, what:type) -> bool:
        return Option[what] == self._type

@dataclass
class ValueProvider[T](Provider[T]):
    _value:T
    _owner:RequestState

    def __init__(self, typ:type[T], value:T, owner:RequestState):
        super().__init__(_type=typ)
        self._value=value
        self._owner=owner

    @override
    def execute(self, state: RequestState) -> T:
        return self._value

@dataclass
class _AggregatorState[T]:
    _generator:Generator[T, None, None]
    _values:list[T] = field(default_factory=list)
    _finished:bool=False

    def new_iter(self) -> Generator[T, None, None]:
        logger = logging.getLogger("ing_api").getChild("AggrState")
        logger.debug(f"Yielding {len(self._values)} values")
        yield from self._values
        logger.debug(f"Done yielding {len(self._values)} values.")
        if not self._finished:
            logger.debug("Generator not finished, continuing")
            for k in self._generator:
                self._values.append(k)
                yield k
            self._finished = True

    def is_finished(self) -> bool:
        return self._finished

@dataclass
class _AggregatorPromise[T]:
    _provider: GeneratorProvider[T]
    _state_map:weakref.WeakKeyDictionary[RequestState, _AggregatorState[T]] = field(default_factory=weakref.WeakKeyDictionary)

    def new_iter(self, caller:RequestState, parents:list[RequestState], use_old:bool) -> Generator[T, None, None]:
        logger = logging.getLogger("ing_api").getChild("AggrProm")
        if caller in self._state_map:
            logger.debug("Yielding from existing aggregator")
            yield from self._state_map[caller].new_iter()
        elif use_old:
            logger.debug(f"Trying {len(parents)} parents")
            for ix, parent in enumerate(parents):
                if parent in self._state_map:
                    logger.debug(f"Yielding from owner's existing aggregator (depth: {ix+1})")
                    yield from self._state_map[parent].new_iter()
                    return
        logger.debug("Creating new aggregator")
        new_aggregator = _AggregatorState(_generator=self._provider.execute(caller))
        self._state_map[caller] = new_aggregator
        yield from self._state_map[caller].new_iter()

    def move_from(self, child_state:RequestState, parent_state:RequestState, overwrite:bool=False, copy:bool=False):
        if child_state not in self._state_map:
            return
        if parent_state in self._state_map and not overwrite:
            if (self._state_map[child_state] != self._state_map[parent_state]):
                raise ValueError("Child state is not allowed to overwrite parent state!")
            return
        self._state_map[parent_state] = self._state_map[child_state]
        if not copy:
            _ = self._state_map.pop(child_state)

    def clone(self) -> _AggregatorPromise[T]:
        return _AggregatorPromise(_provider=self._provider)

@dataclass
class _StorePromise[T]:
    """
    A wrapper object for a Provider that may or may not have
    been called. Use .get() to either get the cached result,
    or to execute the provider and cache the result.
    """
    _provider: Provider[T] | OptionalProvider[T]
    _state_map:weakref.WeakKeyDictionary[RequestState, Option[T]] = field(default_factory=weakref.WeakKeyDictionary)

    def set(self, state:RequestState, val:Option[T]):
        self._state_map[state] = val

    def is_available(self, state:RequestState) -> bool:
        return state in self._state_map

    def get(self, caller: RequestState, allow_deferred:bool, parents:list[RequestState], use_old:bool) -> Option[T]:
        logger = logging.getLogger("ing_api").getChild("StoreProm")
        if caller in self._state_map:
            return self._state_map[caller]
        elif use_old:
            logger.log(5, f"Trying {len(parents)} parents")
            for ix, parent in enumerate(parents):
                if parent in self._state_map:
                    logger.log(5, f"Value found (depth {ix+1})")
                    return self._state_map[parent]
        if self._provider.deferred() and not allow_deferred:
            return Option.none()
        if isinstance(self._provider, OptionalProvider):
            res = cast(Option[T], self._provider(caller))
        else:
            res = Option.some(self._provider(caller))
        self._state_map[caller] = res
        return res

    def provider(self) -> Provider[T] | OptionalProvider[T] | None:
        return self._provider

    def move_from(self, child_state:RequestState, parent_state:RequestState, overwrite:bool=False, copy:bool=False):
        if child_state not in self._state_map:
            return
        if parent_state in self._state_map and not overwrite:
            if (self._state_map[child_state] != self._state_map[parent_state]):
                raise ValueError("Child state is not allowed to overwrite parent state!")
            return
        self._state_map[parent_state] = self._state_map[child_state]
        if not copy:
            _ = self._state_map.pop(child_state)

    def clone(self) -> _StorePromise[T]:
        return _StorePromise(_provider=self._provider)

class RootState:
    _store: dict[type[Any], list[_StorePromise[Any]|_AggregatorPromise[Any]]]
    _lock: RLock

    def __init__(self):
        self._store = {}
        self._lock = RLock()

    def set[T](self, key: type[T], item: T, caller:RequestState, insert:bool=True):
        with self._lock:
            origin = get_origin(key) or key
            if not isinstance(item, origin):
                raise TypeError("Item is not instance of key class")
            self.add_provider(key, ValueProvider[T](typ=origin, value=item, owner=caller), insert)

    def add_provider[T](
        self,
        key: type[T],
        callback: Provider[T] | OptionalProvider[T] | GeneratorProvider[T],
        insert:bool=False
    ):
        logger = logging.getLogger("ingr_api").getChild("RootState").getChild("add_prov")
        with self._lock:
            origin = get_origin(key) or key
            if not callback.provides(origin):
                raise TypeError("Provider does not provide specified key")
            if isinstance(callback, GeneratorProvider):
                prom:_StorePromise[T]|_AggregatorPromise[T] = _AggregatorPromise(_provider=callback)
            else:
                prom = _StorePromise(_provider=callback)

            lst:list[_StorePromise[T]|_AggregatorPromise[T]] = cast(list[_StorePromise[T]|_AggregatorPromise[T]], self._store.setdefault(key, []))
            if not insert:
                logger.log(5, f"Appending provider for {key} (origin: {origin})")
                lst.append(prom)
            else:
                logger.log(5, f"Inserting provider for {key} (origin: {origin})")
                lst.insert(0, prom)

    def iter[T](self, what:type[T], resolve_deferred:bool, caller:RequestState, use_old:bool=True) -> Generator[T, None, None]:
        logger = logging.getLogger("ingr_api").getChild("RootState").getChild("iter")
        parents = caller.get_parents()
        with self._lock:
            if what in self._store:
                lst:list[_StorePromise[T]|_AggregatorPromise[T]] = cast(list[_StorePromise[T]|_AggregatorPromise[T]], self._store[what])
                logger.log(5, f"Found {len(lst)} promises for {what}, yielding (state depth: {len(parents)})...")
                for prom in lst:
                    if isinstance(prom, _AggregatorPromise):
                        yield from prom.new_iter(caller, parents, use_old)
                    else:
                        res = prom.get(caller, resolve_deferred, parents, use_old)
                        if res.is_some():
                            yield res.unwrap()
            else:
                logger.log(5, f"Found no promises for {what}.")

    def move_from(self, from_state:RequestState, to_state:RequestState, overwrite:bool=True, copy:bool=False):
        with self._lock:
            for lst in self._store.values():
                for it in lst:
                    it.move_from(from_state, to_state, overwrite, copy)

    def clone(self) -> RootState:
        new_store:dict[type[Any], list[_StorePromise[Any]|_AggregatorPromise[Any]]] = {}
        for k, lst in self._store.items():
            new_store[k] = [prom.clone() for prom in lst]
        state=RootState()
        state._store = new_store
        return state

    def new_request(self) -> RequestState:
        state=RequestState(root=self)
        return state


class RequestState:
    _root:RootState
    _child: weakref.ReferenceType[RequestState] | None
    _parent: RequestState | None = None
    _poisoned: bool = False

    def get_parents(self) -> list[RequestState]:
        if self._parent is not None:
            return [self._parent] + self._parent.get_parents()
        else:
            return []

    def __init__(self, root:RootState):
        self._child = None
        self._root = root

    def set[T](self, key: type[T], item: T, insert:bool=True):
        self.assert_valid()
        self.assert_not_masked()
        self._root.set(key, item, self, insert)

    def iter[T](self, what:type[T], resolve_deferred:bool=True, use_old:bool=True) -> Generator[T, None, None]:
        self.assert_valid()
        return self._root.iter(what, resolve_deferred, self, use_old)

    def get_optional[T](self, what: type[T], resolve_deferred:bool=True) -> Option[T]:
        self.assert_valid()
        nxt = next(self.iter(what, resolve_deferred), None)
        return Option.some(nxt) if nxt is not None else Option.none()

    def invalidate(self):
        self._poisoned = True
        if self._parent is not None:
            super_msk = None if self._parent._child is None else self._parent._child()
            if super_msk != self:
                raise ValueError("Invalid state: masker has been modified")
            else:
                self._parent._child = None
                self._parent = None

    def is_poisoned(self) -> bool:
        poison: bool = self._poisoned or (
            self._parent is not None and self._parent.is_poisoned()
        )
        if poison:
            self._poisoned = True
        return poison

    def is_masked(self) -> bool:
        msk = None if self._child is None else self._child()
        return msk is not None and not msk.is_poisoned()

    def assert_valid(self):
        if self.is_poisoned():
            raise ValueError("State is poisoned")

    def assert_not_masked(self):
        if self.is_masked():
            raise ValueError("State is being masked and cannot be modified.")

    def commit(self, overwrite:bool=False):
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
        self.assert_valid()
        if self.is_masked():
            raise ValueError(
                "Called commit() on a masked RequestState. Needs to be called on the masking state."
            )
        if self._parent is None:
            raise ValueError("Called commit() on a non-masking RequestState")
        self._root.move_from(self, self._parent, overwrite)
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
        self.assert_valid()
        logger.log(5, "Mask called")
        if not invalidate_previous:
            self.assert_not_masked()
        elif self._child is not None:
            msk = self._child()
            if msk is not None:
                msk.invalidate()
        new_state = RequestState(self._root)
        new_state._parent = self
        self._child = weakref.ref(new_state)
        return new_state

    def get[T](self, what: type[T], allow_deferred:bool=False) -> T:
        self.assert_valid()
        nxt = next(self.iter(what, allow_deferred), None)
        if nxt is None:
            raise ValueError(f"State contains no values of type {what}!")
        return nxt

    def get_all[T](self, what: type[T], allow_deferred:bool=False) -> list[T]:
        self.assert_valid()
        return [k for k in self.iter(what, allow_deferred)]
