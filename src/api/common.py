from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class Option[T]:
    _value: T | None
    _some: bool

    @staticmethod
    def some[K](_val: K) -> Option[K]:
        return Option(_val, _some=True)

    @staticmethod
    def some_if[K](_val: K|None, predicate:Callable[[K], bool]|None = None) -> Option[K]:
        if _val is not None and (predicate is None or predicate(_val)):
            return Option.some(_val)
        else:
            return Option.none()

    @staticmethod
    def none[K]() -> Option[K]:
        return Option(None, _some=False)

    @staticmethod
    def from_union[T1, T2, O](left_type:type[T1], val:T1|T2, map_fn_1:Callable[[T1], O], map_fn_2:Callable[[T2], O]) -> Option[O]:
        if isinstance(val, left_type):
            return Option.some(map_fn_1(val))
        else:
            return Option.some(map_fn_2(cast(T2, val)))

    def unwrap(self) -> T:
        if self._value is None or not self._some:
            raise ValueError("Called .unwrap() on a None value!")
        return self._value

    def unwrap_or(self, default_value:T) -> T:
        if self._value is None or not self._some:
            return default_value
        return self._value

    def to_union(self) -> T|None:
        return self._value

    def unwrap_or_union[V](self, default_value:V) -> T|V:
        if self._value is None or not self._some:
            return default_value
        return self._value

    def expect(self, msg: str) -> T:
        if self._value is None or not self._some:
            raise ValueError(msg)
        return self._value

    def is_some(self) -> bool:
        return self._some

    def is_none(self) -> bool:
        return not self.is_some()

    def map[O](self, map_fn: Callable[[T], O]) -> Option[O]:
        if self.is_some():
            return Option.some(map_fn(self.unwrap()))
        else:
            return Option.none()

    def replace(self, new_val:T) -> Option[T]:
        if self.is_some():
            self._value=new_val
        return self

    def set(self, new_val:T) -> Option[T]:
        self._value=new_val
        self._some = True
        return self

    def inspect(self, inspect_fn:Callable[[T], Any]) -> Option[T]:
        if self.is_some():
            inspect_fn(self.unwrap())
        return self

    def and_then[O](self, map_fn: Callable[[T], Option[O]]) -> Option[O]:
        if self.is_some():
            return map_fn(self.unwrap())
        else:
            return Option.none()

    def and_then_union[O](self, map_fn: Callable[[T], O|None]) -> Option[O]:
        if self.is_some():
            return Option.some_if(map_fn(self.unwrap()))
        else:
            return Option.none()

    def filter(self, predicate:Callable[[T], bool]) -> Option[T]:
        if self.is_some() and predicate(self.unwrap()):
            return self
        return Option.none()

    def get_or(self, other: Option[T]) -> Option[T]:
        if self.is_some():
            return self
        else:
            return other

    def is_none_or(self, predicate: Callable[[T], bool]) -> bool:
        if self.is_none():
            return True
        else:
            return predicate(self.unwrap())

    def is_some_and(self, predicate: Callable[[T], bool]) -> bool:
        if self.is_none():
            return False
        else:
            return predicate(self.unwrap())

@dataclass
class Result[T, E:Exception]:
    _value:T|E
    _err:bool

    @staticmethod
    def ok[K, X:Exception](val:K) -> Result[K, X]:
        return Result(_value=val, _err=False)

    @staticmethod
    def err[K, X:Exception](val:X) -> Result[K, X]:
        return Result(_value=val, _err=True)

    @staticmethod
    def catch[K, X:Exception](catch_what:type[X], lmda: Callable[[], K]) -> Result[K, X]:
        try:
            res:K = lmda()
            return Result.ok(res)
        except catch_what as e:
            return Result.err(e)

    def unwrap(self) -> T:
        if self.is_err():
            raise ValueError("Called unwrap() on Result.err")
        return cast(T, self._value)

    def unwrap_or(self, def_val:T) -> T:
        if self.is_err():
            return def_val
        return self.unwrap()

    def union(self) -> T|E:
        return self._value

    def ok_or_none(self) -> Option[T]:
        if self.is_ok():
            return Option.some(cast(T, self._value))
        else:
            return Option.none()

    def is_ok(self) -> bool:
        return not self._err

    def is_err(self) -> bool:
        return self._err
