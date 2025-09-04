class Axis:
    name: str
    shard_dim: str

    def __init__(self, name: str, shard_dim: str = '') -> None:
        self.name = name
        self.shard_dim = shard_dim

    def local(self) -> bool:
        return not bool(self.shard_dim)

    def __repr__(self):
        if self.local():
            return self.name
        else:
            return f"{self.name} / {self.shard_dim}"

    def __hash__(self):
        return hash(self.name) ^ hash(self.shard_dim)

    def __eq__(self, other: object):
        if not isinstance(other, Axis):
            return NotImplemented

        return self.name == other.name and self.shard_dim == other.shard_dim

# TODO: eventually add replication, so this won't be just a list
class Axes(list[Axis]):
    def __repr__(self):
        return " × ".join(str(x) for x in self)

    def local(self):
        for x in self:
            if not x.local():
                return False
        return True
