# Optimization Policies

`einshard` accepts an optional optimization policy for symbolic plan ranking
diagnostics:

```python
z = es.einshard("a b/tp, b c -> a/tp c", x, w, mesh=mesh, optimize="memory")
```

Available named modes are:

- `training`: default; scores combined forward and backward costs.
- `inference`: ignores backward communication in the cost score.
- `memory`: weights peak and materialized local tensor size more heavily.
- `communication`: weights estimated communicated bytes more heavily.
- `latency`: weights collective count more heavily.

For explicit weights, pass a `PlanPolicy` object:

```python
policy = es.PlanPolicy.from_mode("communication")
z = es.einshard("a b/tp, b c -> a/tp c", x, w, mesh=mesh, policy=policy)
```

Pass either `optimize=` or `policy=`, not both.

Existing call sites can use a scoped default policy:

```python
with es.optimize("memory"):
    z = es.einshard("a b/tp, b c -> a/tp c", x, w, mesh=mesh)
```

Or set a process default:

```python
es.set_default_policy("communication")
try:
    z = es.einshard("a b/tp, b c -> a/tp c", x, w, mesh=mesh)
finally:
    es.set_default_policy(None)
```

Policy precedence is explicit `policy=`, explicit `optimize=`, scoped
`with es.optimize(...)`, process default, then the library default `training`
policy.

Current policies affect cost rankings and inspection snapshots for alternatives
that can be compared safely. Execution remains behavior-preserving for paths
whose alternatives need additional runtime validation or per-alternative shape
metadata.
