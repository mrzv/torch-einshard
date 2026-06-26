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
        self.mesh_dim_names = _mesh_dim_names(device_mesh.mesh_dim_names)
        if len(self.mesh_dim_names) != self.mesh.dim():
            raise ValueError("mesh_dim_names length must match mesh rank")
        self._name_to_dim = {name: i for i, name in enumerate(self.mesh_dim_names)}
        self._compound_groups = {}

    def __getitem__(self, name):
        if name in self._name_to_dim:
            return self.device_mesh[name]
        if "-" not in name:
            return self.device_mesh[name]

        dim_names = tuple(name.split("-"))
        if any(not dim_name for dim_name in dim_names):
            raise ValueError("Compound mesh groups cannot contain empty mesh dimensions")
        if len(set(dim_names)) != len(dim_names):
            raise ValueError("Compound mesh groups cannot repeat mesh dimensions")
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


def _mesh_dim_names(names):
    if names is None:
        return ()
    if isinstance(names, str):
        result = (names,)
    elif isinstance(names, torch.Tensor) or (hasattr(names, "shape") and hasattr(names, "dtype")):
        raise TypeError("mesh_dim_names must be an iterable of strings")
    else:
        try:
            result = tuple(names)
        except TypeError as error:
            raise TypeError("mesh_dim_names must be an iterable of strings") from error
    if any(not isinstance(name, str) for name in result):
        raise TypeError("mesh_dim_names entries must be strings")
    if any(not name for name in result):
        raise ValueError("mesh_dim_names entries must be non-empty strings")
    for name in result:
        components = tuple(name.split("-"))
        if any(not component for component in components):
            raise ValueError("mesh_dim_names entries cannot contain empty compound components")
        if len(set(components)) != len(components):
            raise ValueError("mesh_dim_names entries cannot repeat compound components")
    seen_components = set()
    for name in result:
        components = set(name.split("-"))
        if seen_components.intersection(components):
            raise ValueError("mesh_dim_names entries cannot overlap compound components")
        seen_components.update(components)
    if len(set(result)) != len(result):
        raise ValueError("mesh_dim_names entries must be unique")
    return result
