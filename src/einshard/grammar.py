from parsley import makeGrammar

# TODO: add subscript-notation for parallelism and ellipsis
# TODO: for now only two tensors
grammar = r"""
    ws = (' ' | '\r' | '\n' | '\t')*

    id = <letterOrDigit+>
    sharded = id:a ws '/' ws id:p -> (a,p)
    axes = (ws (sharded | id))+

    map = axes:a ws (',' axes:x ws -> x)?:aa '->' axes:o -> (a, aa, o)
"""

sharding = makeGrammar(grammar, globals(), name = "Einshard")

if __name__ == '__main__':
    print(sharding('a b c').axes())
    print(sharding('a b / dp c').axes())
    print(sharding('a b / dp c, a c -> b/dp').map())
    print(sharding('b c -> b/dp c').map())
