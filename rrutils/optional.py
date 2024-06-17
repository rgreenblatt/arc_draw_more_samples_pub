from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")

# damn, rust is so much cooler


def it(x: Optional[T]) -> Iterable[T]:
    if x is not None:
        yield x


def map(x: Optional[T], f: Callable[[T], U]) -> Optional[U]:
    if x is not None:
        return f(x)
    return None


def unwrap(x: Optional[T], error: str = "") -> T:
    assert x is not None, error
    return x


def unwrap_or(x: Optional[T], default: T) -> T:
    if x is None:
        return default
    return x


def unwrap_or_else(x: Optional[T], default: Callable[[], T]) -> T:
    if x is None:
        return default()
    return x


def or_y(x: Optional[T], y: Optional[T]) -> Optional[T]:
    if x is None:
        return y
    return x
