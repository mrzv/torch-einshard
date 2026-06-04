import re


_FAMILY = re.compile(r"\*([A-Za-z0-9]+)")
_ZIPPED = re.compile(r"\[([^\]]+)\]")


def _normalize_families(families):
    if families is None:
        return {}
    return {name: tuple(axes) for name, axes in families.items()}


def _expand_sizes(sizes, families):
    if not isinstance(sizes, dict) or not families:
        return sizes

    result = {}
    for key, value in sizes.items():
        if key not in families:
            result[key] = value
            continue

        axes = families[key]
        if len(value) != len(axes):
            raise ValueError(f"Size family {key!r} has {len(value)} values for {len(axes)} axes")
        for axis, axis_size in zip(axes, value):
            if axis in result and result[axis] != axis_size:
                raise ValueError(f"Conflicting sizes for expanded axis {axis!r}")
            result[axis] = axis_size
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
    return expression, _expand_sizes(sizes, families)
