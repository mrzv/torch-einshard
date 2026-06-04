from itertools import product

import torch
import torch.distributed as dist


class _MeshDim:
    def __init__(self, group):
        self.group = group

    def get_group(self):
        return self.group


class CompoundDeviceMesh:
    def __init__(self, device_mesh):
        self.device_mesh = device_mesh
        self.mesh = device_mesh.mesh
        self.mesh_dim_names = tuple(device_mesh.mesh_dim_names)
        self._name_to_dim = {name: i for i, name in enumerate(self.mesh_dim_names)}
        self._compound_groups = {}

    def __getitem__(self, name):
        if name in self._name_to_dim:
            return self.device_mesh[name]
        if "-" not in name:
            return self.device_mesh[name]

        dim_names = tuple(name.split("-"))
        if any(dim_name not in self._name_to_dim for dim_name in dim_names):
            return self.device_mesh[name]
        return _MeshDim(self._compound_group(dim_names))

    def _compound_group(self, dim_names):
        dim_names = tuple(sorted(dim_names, key=self._name_to_dim.__getitem__))
        if dim_names in self._compound_groups:
            return self._compound_groups[dim_names]
        if not dist.is_initialized():
            raise RuntimeError("Compound mesh groups require an initialized process group")

        compound_dims = {self._name_to_dim[name] for name in dim_names}
        fixed_dims = [dim for dim in range(self.mesh.dim()) if dim not in compound_dims]
        fixed_ranges = [range(self.mesh.shape[dim]) for dim in fixed_dims]
        current_rank = dist.get_rank()
        current_group = None
        mesh = self.mesh.detach().cpu()

        for fixed_index in product(*fixed_ranges) if fixed_ranges else [()]:
            ranks = []
            for index in product(*(range(size) for size in mesh.shape)):
                if all(index[dim] == fixed_index[i] for i, dim in enumerate(fixed_dims)):
                    ranks.append(int(mesh[index]))
            group = dist.new_group(ranks=ranks)
            if current_rank in ranks:
                current_group = group

        self._compound_groups[dim_names] = current_group
        return current_group


def wrap_mesh(device_mesh):
    if isinstance(device_mesh, CompoundDeviceMesh):
        return device_mesh
    return CompoundDeviceMesh(device_mesh)
