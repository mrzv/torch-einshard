from copy import deepcopy
from functools import lru_cache

from parsley import makeGrammar

from .sharding import Axis, Axes, AxisGroup, EllipsisAxis, TensorSpec

# TODO: add subscript-notation for parallelism
# TODO: for now only two tensors
grammar = r"""
    ws = (' ' | '\r' | '\n' | '\t')*

    id = <letterOrDigit+>
    mesh_id = <letterOrDigit+ ('-' letterOrDigit+)*>
    sharded = id:a ws '/' ws mesh_id:p -> Axis(a,p)
    ellipsis = '...' -> EllipsisAxis()
    axis = sharded | id:a -> Axis(a)
    group = '(' (ws axis)+:axs ws ')' -> AxisGroup(axs)
    axes = (ws (group | ellipsis | axis))+:axs -> Axes(axs)
    partial_many = '(' ws mesh_id:first (ws ',' ws mesh_id)*:rest ws ')' -> [first] + rest
    partial = ws '//' ws (partial_many | mesh_id:p -> [p])
    tensor = axes:a (partial:p -> p)?:p -> TensorSpec(a, p or [])

    map = tensor:a ws (',' tensor:x ws -> x)?:aa '->' tensor:o -> (a, aa, o)
"""

sharding = makeGrammar(grammar, globals(), name = "Einshard")


def parse_sharding(expression):
    try:
        return deepcopy(_parse_sharding_cached(expression))
    except Exception as error:
        raise ValueError(f"Invalid einshard expression {expression!r}: {error}") from error


@lru_cache(maxsize=2048)
def _parse_sharding_cached(expression):
    return sharding(expression).map()
