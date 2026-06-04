import re
from functools import lru_cache


_FAMILY = re.compile(r"\*([A-Za-z0-9]+)")
_ZIPPED = re.compile(r"\[([^\]]+)\]")


def _normalize_families(families):
    if families is None:
        return {}
    return {name: tuple(axes) for name, axes in families.items()}


def _family_key(families):
    families = _normalize_families(families)
    return tuple(sorted(families.items()))


def _mapping_key(mapping):
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        return mapping
    return tuple(sorted(
        (key, tuple(value) if isinstance(value, (list, tuple)) else value)
        for key, value in mapping.items()
    ))


def _mapping_from_key(key):
    if key is None or isinstance(key, list):
        return key
    if not isinstance(key, tuple):
        return key
    return {item_key: item_value for item_key, item_value in key}


def _axis_name(axis):
    return axis.split("/", 1)[0].strip()


def expand_family_mapping(mapping, families, *, label):
    families = _normalize_families(families)
    if not isinstance(mapping, dict) or not families:
        return mapping

    result = {}
    for key, value in mapping.items():
        if key not in families:
            result[key] = value
            continue

        axes = families[key]
        if isinstance(value, (str, bytes)):
            values = (value,) * len(axes)
        else:
            try:
                values = tuple(value)
            except TypeError:
                values = (value,) * len(axes)
        if len(values) != len(axes):
            raise ValueError(f"{label} family {key!r} has {len(values)} values for {len(axes)} axes")
        for axis, axis_value in zip(axes, values):
            axis = _axis_name(axis)
            if axis in result and result[axis] != axis_value:
                raise ValueError(f"Conflicting {label.lower()} for expanded axis {axis!r}")
            result[axis] = axis_value
    return result


def _expand_zipped(match, families):
    tokens = match.group(1).split()
    if not tokens or any(not token.startswith("*") for token in tokens):
        raise ValueError("Zipped axis-family groups must contain only family references")

    names = [token[1:] for token in tokens]
    missing = [name for name in names if name not in families]
    if missing:
        raise ValueError(f"Unknown axis family {missing[0]!r}")

    lengths = {len(families[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("Zipped axis families must have matching lengths")

    return " ".join(
        f"({' '.join(axes)})"
        for axes in zip(*(families[name] for name in names))
    )


def expand_axis_families(expression, sizes=None, families=None):
    families = _normalize_families(families)
    if not families:
        return expression, sizes

    def expand_family(match):
        name = match.group(1)
        if name not in families:
            raise ValueError(f"Unknown axis family {name!r}")
        return " ".join(families[name])

    expression = _ZIPPED.sub(lambda match: _expand_zipped(match, families), expression)
    expression = _FAMILY.sub(expand_family, expression)
    return expression, expand_family_mapping(sizes, families, label="Size")


@lru_cache(maxsize=1024)
def _cached_expand_axis_families(expression, sizes_key, families_key):
    return expand_axis_families(expression, _mapping_from_key(sizes_key), dict(families_key))


def cached_expand_axis_families(expression, sizes=None, families=None):
    return _cached_expand_axis_families(expression, _mapping_key(sizes), _family_key(families))
