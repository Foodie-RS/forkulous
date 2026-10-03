from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
