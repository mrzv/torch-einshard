from einshard.grammar import sharding

print(sharding('a b c').axes())
print(sharding('a b / dp c').axes())
print(sharding('a b / dp c, a c -> b/dp').map())
print(sharding('b c -> b/dp c').map())
