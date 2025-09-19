from parsley import makeGrammar
from .sharding import Axis, Axes

# TODO: add subscript-notation for parallelism and ellipsis
# TODO: for now only two tensors
grammar = r"""
    ws = (' ' | '\r' | '\n' | '\t')*

    id = <letterOrDigit+>
    sharded = id:a ws '/' ws id:p -> Axis(a,p)
    axes = (ws (sharded | id:a -> Axis(a)))+:axs -> Axes(axs)

    map = axes:a ws (',' axes:x ws -> x)?:aa '->' axes:o -> (a, aa, o)
"""

sharding = makeGrammar(grammar, globals(), name = "Einshard")
