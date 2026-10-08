from __future__ import annotations

import abc
from enum import Enum
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

@dataclass
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

    def new_iter(self, only_use_available:bool, continue_from:T|None) -> Generator[T, None, None]:
        logger = logging.getLogger("ing_api").getChild("AggrState")
        if continue_from is not None and continue_from not in self._values:
            logger.log(5, "Not yielding anything, as continue_from was not found in values.")
            return
        elif continue_from is not None:
            ix = self._values.index(continue_from)
            logger.log(5, "Continue value found. Continuing.")
            yield from self._values[ix:]

        logger.debug(f"Yielding {len(self._values)} values")
        yield from self._values
        logger.debug(f"Done yielding {len(self._values)} values.")
        if only_use_available and not self._finished:
            logger.log(5, "Generator not exhausted, but only_use_available is true.")
            return
        if not self._finished:
            logger.log(5, "Generator not finished, continuing")
            for k in self._generator:
                self._values.append(k)
                yield k
            self._finished = True
        else:
            logger.log(5, "Generator already finished.")

    def is_finished(self) -> bool:
        return self._finished

    def available(self) -> int:
        return len(self._values)

@dataclass
class _AggregatorPromise[T]:
    _provider: GeneratorProvider[T]
    _state_map:weakref.WeakKeyDictionary[RequestState, _AggregatorState[T]] = field(default_factory=weakref.WeakKeyDictionary)

    def new_iter(self, caller:RequestState, parents:list[RequestState], use_old:bool, only_use_available:bool, continue_from:T|None) -> Generator[T, None, None]:
        logger = logging.getLogger("ing_api").getChild("AggrProm")
        if caller in self._state_map:
            logger.log(5, f"Yielding from existing aggregator for {self._provider._type}")
            yield from self._state_map[caller].new_iter(only_use_available=only_use_available, continue_from=continue_from)
        elif use_old:
            logger.debug(f"Trying {len(parents)} parents")
            for ix, parent in enumerate(parents):
                if parent in self._state_map:
                    logger.log(5, f"Yielding from owner's existing aggregator (depth: {ix+1})")
                    yield from self._state_map[parent].new_iter(only_use_available=only_use_available, continue_from=continue_from)
                    return
        if only_use_available:
            logger.log(5, "Requested only available values, so not creating a new aggregator")
            return
        if continue_from is not None:
            logger.log(5, "Not creating new aggregator, as continue_after is not None and can't be in the new aggregator.")
            return
        logger.log(5, "Creating new aggregator")
        new_aggregator = _AggregatorState(_generator=self._provider.execute(caller))
        self._state_map[caller] = new_aggregator
        yield from self._state_map[caller].new_iter(False, continue_from=continue_from)

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

    def count_available(self, caller:RequestState, parents:list[RequestState], use_old:bool) -> int:
        count = 0
        if caller in self._state_map:
            count += self._state_map[caller].available()
        if use_old:
            for parent in parents:
                if parent in self._state_map:
                    count += self._state_map[parent].available()
        return count

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

    def is_available(self, state:RequestState, parents:list[RequestState], use_old:bool) -> bool:
        return state in self._state_map or (use_old and len(set(self._state_map.keys()).intersection(parents)) > 0)

    def get(self, caller: RequestState, allow_deferred:bool, parents:list[RequestState], use_old:bool, only_available:bool) -> Option[T]:
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
            logger.log(5, "Provider is deferred and allow_deferred is false, so not calling provider.")
            return Option.none()
        if only_available:
            logger.log(5, "Requested only available values. Value is not available.")
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

type FromProviderType[T]=type[Provider[T]|GeneratorProvider[T]|OptionalProvider[T]]|None

class RootState:
    _store: dict[type[Any], list[_StorePromise[Any]|_AggregatorPromise[Any]]]
    _lock: RLock

    def __init__(self):
        self._store = {}
        self._lock = RLock()

    def set[T](self, key: type[T], item: T, caller:RequestState, insert:bool=True):
        # TODO replace set() logic so that it updates a ValueProvider instead of adding a new one
        with self._lock:
            origin = get_origin(key) or key
            if not isinstance(item, origin):
                raise TypeError("Item is not instance of key class")
            self.add_provider(key, ValueProvider[T](typ=origin, value=item, owner=caller), insert)

    def count_available[T](self, caller:RequestState, what:type[T], use_old:bool) -> int:
        if what not in self._store:
            return 0
        count = 0
        with self._lock:
            parents = caller.get_parents()
            for prom in self._store[what]:
                if isinstance(prom, _AggregatorPromise):
                    count += prom.count_available(caller, parents, use_old)
                else:
                    count += 1 if prom.is_available(caller, parents, use_old) else 0
            return count


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

    def iter[T](self, what:type[T], resolve_deferred:bool, caller:RequestState, use_parent:bool, only_use_available:bool, from_provider:FromProviderType[T], continue_after:T|None, round_robin:bool) -> Generator[T, None, None]:
        logger = logging.getLogger("ingr_api").getChild("RootState").getChild("iter")
        if round_robin and (continue_after is not None):
            raise ValueError("Can't use continue_after and round_robin at the same time!")
        parents = caller.get_parents()
        skipping=continue_after is not None
        with self._lock:
            if what in self._store:
                aggregators:list[Generator[T, None, None]] = []
                lst:list[_StorePromise[T]|_AggregatorPromise[T]] = cast(list[_StorePromise[T]|_AggregatorPromise[T]], self._store[what])
                logger.log(5, f"Found {len(lst)} promises for {what}, yielding (state depth: {len(parents)})...")
                for prom in lst:
                    if from_provider is not None and not isinstance(prom._provider, from_provider):
                        logger.log(5, f"Skipping promise from {prom._provider.__class__}, because it is not the requested provider.")
                        continue
                    if isinstance(prom, _AggregatorPromise):
                        logger.log(5, f"Yielding from aggregator over {prom._provider.__class__}")
                        if not skipping:
                            iter = prom.new_iter(caller, parents, use_parent, only_use_available, continue_from=None)
                            if not round_robin:
                                yield from iter
                            else:
                                try:
                                    logger.log(5, "Yielding one from aggregator, because round_robin is true")
                                    nxt = next(iter)
                                    yield nxt
                                    aggregators.append(iter)
                                except StopIteration:
                                    pass
                        else:
                            iter = prom.new_iter(caller, parents, use_parent, only_use_available, continue_from=continue_after)
                            item = next(iter, None)
                            if item is not None:
                                if item != continue_after:
                                    raise ValueError("The item yielded by the generator is not the expected item!")
                                logger.log(5, f"{prom._provider.__class__} provides the item in continue_after. Continuing...")
                                skipping = False
                                yield from iter
                    else:
                        logger.log(5, f"Calling StorePromise over {prom._provider.__class__}")
                        if skipping:
                            res = prom.get(caller, resolve_deferred, parents, use_parent, only_available=True)
                            if res.is_some_and(lambda k:k == continue_after):
                                logger.log(5, "Found continue_after value. Continuing as normal.")
                                skipping = False
                        else:
                            res = prom.get(caller, resolve_deferred, parents, use_parent, only_use_available)
                            if res.is_some():
                                yield res.unwrap()
                while len(aggregators) > 0:
                    logger.log(5, f"{len(aggregators)} aggregators remaining for {what}")
                    rem:list[Generator[T, None, None]] = []
                    for aggr in aggregators:
                        try:
                            nxt = next(aggr)
                            yield nxt
                        except StopIteration:
                            rem.append(aggr)
                    for r in rem:
                        aggregators.remove(r)

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

    def iter[T](self, what:type[T], resolve_deferred:bool=True, use_parent:bool=True, only_use_available:bool=False, from_provider:FromProviderType[T]=None, continue_after:T|None=None, round_robin:bool=False) -> Generator[T, None, None]:
        self.assert_valid()
        return self._root.iter(what, resolve_deferred, self, use_parent=use_parent, only_use_available=only_use_available, from_provider=from_provider, continue_after=continue_after, round_robin=round_robin)

    def count_available[T](self, what:type[T], use_old:bool=True) -> int:
        return self._root.count_available(self, what, use_old)

    def get_optional[T](self, what: type[T], resolve_deferred:bool=True, allow_parent:bool=True, only_use_available:bool=False, from_provider:FromProviderType[T]=None, continue_after:T|None=None) -> Option[T]:
        self.assert_valid()
        nxt = next(self.iter(what, resolve_deferred=resolve_deferred, use_parent=allow_parent, only_use_available=only_use_available, from_provider=from_provider, continue_after=continue_after), None)
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

    def get[T](self, what: type[T], allow_deferred:bool=False, continue_after:T|None=None) -> T:
        self.assert_valid()
        nxt = next(self.iter(what, allow_deferred, continue_after=continue_after), None)
        if nxt is None:
            raise ValueError(f"State contains no values of type {what}!")
        return nxt

    def get_all[T](self, what: type[T], allow_deferred:bool=False, allow_parent:bool=True, only_use_available:bool=False, from_provider:FromProviderType[T]=None) -> list[T]:
        self.assert_valid()
        return [k for k in self.iter(what, allow_deferred, use_parent=allow_parent, only_use_available=only_use_available, from_provider=from_provider)]
