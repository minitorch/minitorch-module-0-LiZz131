"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def id(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """$f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return (x + abs(x)) / 2


EPS = 1e-6


def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""If $f = log$ as above, compute $d \times f'(x)$"""
    return d * inv(x)


def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"""If $f(x) = 1/x$ compute $d \times f'(x)$"""
    return d * -1 / (x * x)


def relu_back(x: float, d: float) -> float:
    r"""If $f = relu$ compute $d \times f'(x)$"""
    return d if x >= 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], lst: Iterable[float]) -> List[float]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.
        lst: Iterable of values.

    Returns:
    -------
        list of values.

    """
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> List[float]:
    """Higher-order zipwith (aka map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values
        lst1: list of values
        lst2: list of values

    Returns:
    -------
        list of values.

    """
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    fn: Callable[[float, float], float], lst: Iterable[float], start: float
) -> float:
    """Higher-order reduce.

    See https://en.wikipedia.org/wiki/Reduce

    Args:
    ----
        fn: combine two values
        lst: list of values
        start: start value $x_0$

    Returns:
    -------
        float value

    """
    result = start
    for x in lst:
        result = fn(result, x)
    return result


def negList(lst: Iterable[float]) -> List[float]:
    """Use `map` and `neg` to negate each element in `lst`."""
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> List[float]:
    """Add the elements of `lst1` and `lst2` using `zipWith` and `add`."""
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    """Take the product of a list using `reduce` and `mul`."""
    return reduce(mul, lst, 1.0)
