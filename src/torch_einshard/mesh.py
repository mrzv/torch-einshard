import torch
import torch.distributed as dist

# Device mesh classes that matches torch DeviceMesh, but adds extra features,
# including groups spanning multiple dimensions
class DeviceMesh:
    def __init__(self, mesh_shape, group, *, mesh_dim_names=None):
        self.group = group
        # FIXME
        # self.rank = dist.
        # self.size = size
        self.shape = mesh_shape
        self.dim_names = mesh_dim_names

        self.names = {}
        if self.dim_names is not None:
            for i,name in enumerate(self.dim_names):
                self.names[name] = i

        self.mesh = torch.arange(self.size).reshape(self.shape)
        self.coordinate = (self.mesh == self.rank).nonzero()[0]

        # setup singleton groups
        self.groups = {}
        
