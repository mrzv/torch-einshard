import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_sharded_contraction_matches_manual_allreduce(dist_env, mesh_2d):
    x = torch.randn(8, 5, requires_grad=True)
    y = torch.randn(5, 10, requires_grad=True)

    z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh_2d)

    expected = es.einshard("al bl, bl c -> al c", x, y)
    expected = es.helpers.all_reduce(expected, mesh_2d["dp"].get_group())
    assert_close(z, expected)


def test_sharded_contraction_matches_gathered_reference(dist_env, mesh_2d):
    x = torch.randn(8, 5, requires_grad=True)
    y = torch.randn(5, 10, requires_grad=True)

    z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh_2d)
    loss = z.norm()
    loss.backward()

    xx = x.detach().clone().requires_grad_(True)
    x_all = es.einshard("a/sp b/dp -> a/sp b", xx, mesh=mesh_2d)
    x_all.retain_grad()

    yy = y.detach().clone().requires_grad_(True)
    y_all = es.einshard("b/dp c -> b c", yy, mesh=mesh_2d)

    z_all = es.einshard("a/sp b, b c -> a/sp c", x_all, y_all)
    assert_close(z_all, z)

    z_all.norm().backward()
    assert_close(xx.grad, x.grad)

    split_grad = es.einshard("a/sp b -> a/sp b/dp", x_all.grad.detach().clone(), mesh=mesh_2d)
    assert_close(split_grad, x.grad)


def test_column_parallel_linear_pattern(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    hidden = world_size * 3

    x = torch.randn(2, 4, 5)
    weight = torch.randn(hidden, 5)
    weight_shard = torch.split(weight, hidden // world_size, dim=0)[rank].contiguous()

    z = es.einshard("b n c, h/tp c -> b n h/tp", x, weight_shard, mesh=mesh_tp)

    expected_full = torch.einsum("bnc,hc->bnh", x, weight)
    expected = torch.split(expected_full, hidden // world_size, dim=2)[rank]
    assert_close(z, expected)


def test_row_parallel_linear_pattern(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    channels = world_size * 3

    x = torch.randn(2, 4, channels)
    weight = torch.randn(7, channels)
    x_shard = torch.split(x, channels // world_size, dim=2)[rank].contiguous()
    weight_shard = torch.split(weight, channels // world_size, dim=1)[rank].contiguous()

    z = es.einshard("b n c/tp, h c/tp -> b n h", x_shard, weight_shard, mesh=mesh_tp)

    expected = torch.einsum("bnc,hc->bnh", x, weight)
    assert_close(z, expected)


def test_row_parallel_linear_explicit_partial_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    channels = world_size * 3

    x = torch.randn(2, 4, channels)
    weight = torch.randn(7, channels)
    x_shard = torch.split(x, channels // world_size, dim=2)[rank].contiguous()
    weight_shard = torch.split(weight, channels // world_size, dim=1)[rank].contiguous()

    z = es.einshard("b n c/tp, h c/tp -> b n h // tp", x_shard, weight_shard, mesh=mesh_tp)

    expected = torch.einsum("bnc,hc->bnh", x_shard, weight_shard)
    assert_close(z, expected)


def test_multi_axis_sharded_contraction(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    sp_group = mesh_2d["sp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    sp_size = dist.get_world_size(sp_group)
    rows = dp_size * 2
    cols = sp_size * 3

    x = torch.randn(rows, cols, 5)
    y = torch.randn(rows, cols, 7)
    x_shard = torch.split(torch.split(x, rows // dp_size, dim=0)[dp_rank], cols // sp_size, dim=1)[sp_rank]
    y_shard = torch.split(torch.split(y, rows // dp_size, dim=0)[dp_rank], cols // sp_size, dim=1)[sp_rank]

    z = es.einshard("a/dp b/sp c, a/dp b/sp d -> c d", x_shard, y_shard, mesh=mesh_2d)

    expected = torch.einsum("abc,abd->cd", x, y)
    assert_close(z, expected)


def test_multi_axis_sharded_contraction_explicit_partial_output(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    sp_group = mesh_2d["sp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    sp_size = dist.get_world_size(sp_group)
    rows = dp_size * 2
    cols = sp_size * 3

    x = torch.randn(rows, cols, 5)
    y = torch.randn(rows, cols, 7)
    x_shard = torch.split(torch.split(x, rows // dp_size, dim=0)[dp_rank], cols // sp_size, dim=1)[sp_rank]
    y_shard = torch.split(torch.split(y, rows // dp_size, dim=0)[dp_rank], cols // sp_size, dim=1)[sp_rank]

    z = es.einshard("a/dp b/sp c, a/dp b/sp d -> c d // (dp,sp)", x_shard, y_shard, mesh=mesh_2d)

    expected = torch.einsum("abc,abd->cd", x_shard, y_shard)
    assert_close(z, expected)


def test_cross_sharded_contraction_reduce_scatters_output_axis(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    sp_group = mesh_2d["sp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    sp_size = dist.get_world_size(sp_group)
    rows = sp_size * 2 + 1
    hidden = dp_size * sp_size * 2 + 1
    cols = dp_size * 3 + 2
    row_shapes = es.helpers.compute_split_shapes(rows, sp_size)
    hidden_dp_shapes = es.helpers.compute_split_shapes(hidden, dp_size)
    hidden_sp_shapes = es.helpers.compute_split_shapes(hidden, sp_size)
    col_shapes = es.helpers.compute_split_shapes(cols, dp_size)
    shapes = {
        "dp": {"e": hidden_dp_shapes, "f": col_shapes},
        "sp": {"l": row_shapes, "e": hidden_sp_shapes},
    }

    x_full = torch.randn(rows, hidden)
    y_full = torch.randn(hidden, cols)
    x_shard = torch.split(
        torch.split(x_full, row_shapes, dim=0)[sp_rank],
        hidden_dp_shapes,
        dim=1,
    )[dp_rank].clone().requires_grad_(True)
    y_shard = torch.split(
        torch.split(y_full, hidden_sp_shapes, dim=0)[sp_rank],
        col_shapes,
        dim=1,
    )[dp_rank].clone().requires_grad_(True)

    z = es.einshard(
        "l/sp e/dp, e/sp f/dp -> l/sp f/dp",
        x_shard,
        y_shard,
        mesh=mesh_2d,
        shapes=shapes,
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y_full.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("le,ef->lf", x_ref, y_ref)
    expected = torch.split(
        torch.split(expected_full, row_shapes, dim=0)[sp_rank],
        col_shapes,
        dim=1,
    )[dp_rank]
    assert_close(z, expected)

    grad_full = torch.randn(rows, cols)
    grad_shard = torch.split(
        torch.split(grad_full, row_shapes, dim=0)[sp_rank],
        col_shapes,
        dim=1,
    )[dp_rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    expected_x_grad = torch.split(
        torch.split(x_ref.grad, row_shapes, dim=0)[sp_rank],
        hidden_dp_shapes,
        dim=1,
    )[dp_rank]
    expected_y_grad = torch.split(
        torch.split(y_ref.grad, hidden_sp_shapes, dim=0)[sp_rank],
        col_shapes,
        dim=1,
    )[dp_rank]
    assert_close(x_shard.grad, expected_x_grad)
    assert_close(y_shard.grad, expected_y_grad)


def test_binary_gathers_free_axis_to_sharded_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    cols = world_size * 3 + 2
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)
    col_shapes = es.helpers.compute_split_shapes(cols, world_size)

    x_full = torch.randn(rows, hidden)
    y_full = torch.randn(hidden, cols)
    x_shard = torch.split(x_full, row_shapes, dim=0)[rank].clone().requires_grad_(True)
    y_shard = torch.split(y_full, col_shapes, dim=1)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l/tp e, e f/tp -> l f/tp",
        x_shard,
        y_shard,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes, "f": col_shapes}},
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y_full.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("le,ef->lf", x_ref, y_ref)
    expected = torch.split(expected_full, col_shapes, dim=1)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(rows, cols)
    grad_shard = torch.split(grad_full, col_shapes, dim=1)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(x_shard.grad, torch.split(x_ref.grad, row_shapes, dim=0)[rank])
    assert_close(y_shard.grad, torch.split(y_ref.grad, col_shapes, dim=1)[rank])


def test_binary_gathers_free_axis_to_full_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    cols = 5
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)

    x_full = torch.randn(rows, hidden)
    y = torch.randn(hidden, cols)
    x_shard = torch.split(x_full, row_shapes, dim=0)[rank].contiguous()

    z = es.einshard(
        "l/tp e, e f -> l f",
        x_shard,
        y,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes}},
    )

    expected = torch.einsum("le,ef->lf", x_full, y)
    assert_close(z, expected)


def test_binary_splits_free_axis_to_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    cols = 5
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)

    x = torch.randn(rows, hidden)
    y = torch.randn(hidden, cols)

    z = es.einshard(
        "l e, e f -> l/tp f",
        x,
        y,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes}},
    )

    expected = torch.split(torch.einsum("le,ef->lf", x, y), row_shapes, dim=0)[rank]
    assert_close(z, expected)


def test_binary_reduce_scatters_contraction_to_output_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    cols = world_size * 3 + 2
    hidden = 5
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)
    col_shapes = es.helpers.compute_split_shapes(cols, world_size)

    x_full = torch.randn(rows, cols)
    y_full = torch.randn(cols, hidden)
    x_shard = torch.split(x_full, col_shapes, dim=1)[rank].clone().requires_grad_(True)
    y_shard = torch.split(y_full, col_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l f/tp, f/tp e -> l/tp e",
        x_shard,
        y_shard,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes, "f": col_shapes}},
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y_full.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("lf,fe->le", x_ref, y_ref)
    expected = torch.split(expected_full, row_shapes, dim=0)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(rows, hidden)
    grad_shard = torch.split(grad_full, row_shapes, dim=0)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(x_shard.grad, torch.split(x_ref.grad, col_shapes, dim=1)[rank])
    assert_close(y_shard.grad, torch.split(y_ref.grad, col_shapes, dim=0)[rank])


def test_binary_splits_full_contracted_axis_to_match_shard(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = 5
    cols = world_size * 3 + 2
    hidden = 4
    col_shapes = es.helpers.compute_split_shapes(cols, world_size)

    x = torch.randn(rows, cols, requires_grad=True)
    y_full = torch.randn(cols, hidden)
    y_shard = torch.split(y_full, col_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l f, f/tp e -> l e",
        x,
        y_shard,
        mesh=mesh_tp,
        shapes={"tp": {"f": col_shapes}},
    )

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y_full.detach().clone().requires_grad_(True)
    expected = torch.einsum("lf,fe->le", x_ref, y_ref)
    assert_close(z, expected)

    grad = torch.randn(rows, hidden)
    z.backward(grad)
    expected.backward(grad)

    assert_close(x.grad, x_ref.grad)
    assert_close(y_shard.grad, torch.split(y_ref.grad, col_shapes, dim=0)[rank])


def test_binary_repartitions_free_axis_after_local_contraction(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    cols = world_size * 3 + 2
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)
    col_shapes = es.helpers.compute_split_shapes(cols, world_size)

    x_full = torch.randn(rows, hidden)
    y = torch.randn(hidden, cols, requires_grad=True)
    x_shard = torch.split(x_full, row_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l/tp e, e f -> l f/tp",
        x_shard,
        y,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes, "f": col_shapes}},
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("le,ef->lf", x_ref, y_ref)
    expected = torch.split(expected_full, col_shapes, dim=1)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(rows, cols)
    grad_shard = torch.split(grad_full, col_shapes, dim=1)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(x_shard.grad, torch.split(x_ref.grad, row_shapes, dim=0)[rank])
    assert_close(y.grad, y_ref.grad)


def test_binary_elementwise_shared_sharded_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)

    x_full = torch.randn(rows, hidden)
    y_full = torch.randn(rows, hidden)
    x_shard = torch.split(x_full, row_shapes, dim=0)[rank].clone().requires_grad_(True)
    y_shard = torch.split(y_full, row_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l/tp e, l/tp e -> l/tp e",
        x_shard,
        y_shard,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes}},
    )

    expected = torch.split(x_full * y_full, row_shapes, dim=0)[rank]
    assert_close(z, expected)


def test_binary_elementwise_splits_replicated_shared_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2 + 1
    hidden = 4
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)

    x_full = torch.randn(rows, hidden)
    y = torch.randn(rows, hidden, requires_grad=True)
    x_shard = torch.split(x_full, row_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "l/tp e, l e -> l/tp e",
        x_shard,
        y,
        mesh=mesh_tp,
        shapes={"tp": {"l": row_shapes}},
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    expected_full = x_ref * y_ref
    expected = torch.split(expected_full, row_shapes, dim=0)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(rows, hidden)
    grad_shard = torch.split(grad_full, row_shapes, dim=0)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(x_shard.grad, torch.split(x_ref.grad, row_shapes, dim=0)[rank])
    assert_close(y.grad, y_ref.grad)


def test_attention_scores_gather_key_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    batch = 2
    heads = 3
    query = world_size * 2 + 1
    key = world_size * 3 + 2
    dim = 4
    query_shapes = es.helpers.compute_split_shapes(query, world_size)
    key_shapes = es.helpers.compute_split_shapes(key, world_size)

    q_full = torch.randn(batch, heads, query, dim)
    k_full = torch.randn(batch, heads, key, dim)
    q_shard = torch.split(q_full, query_shapes, dim=2)[rank].clone().requires_grad_(True)
    k_shard = torch.split(k_full, key_shapes, dim=2)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "b h q/tp d, b h k/tp d -> b h q/tp k",
        q_shard,
        k_shard,
        mesh=mesh_tp,
        shapes={"tp": {"q": query_shapes, "k": key_shapes}},
    )

    q_ref = q_full.detach().clone().requires_grad_(True)
    k_ref = k_full.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("bhqd,bhkd->bhqk", q_ref, k_ref)
    expected = torch.split(expected_full, query_shapes, dim=2)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(batch, heads, query, key)
    grad_shard = torch.split(grad_full, query_shapes, dim=2)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(q_shard.grad, torch.split(q_ref.grad, query_shapes, dim=2)[rank])
    assert_close(k_shard.grad, torch.split(k_ref.grad, key_shapes, dim=2)[rank])


def test_attention_values_split_contracted_key_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    batch = 2
    heads = 3
    query = world_size * 2 + 1
    key = world_size * 3 + 2
    dim = 4
    query_shapes = es.helpers.compute_split_shapes(query, world_size)
    key_shapes = es.helpers.compute_split_shapes(key, world_size)

    attn_full = torch.randn(batch, heads, query, key)
    v_full = torch.randn(batch, heads, key, dim)
    attn_shard = torch.split(attn_full, query_shapes, dim=2)[rank].clone().requires_grad_(True)
    v_shard = torch.split(v_full, key_shapes, dim=2)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "b h q/tp k, b h k/tp d -> b h q/tp d",
        attn_shard,
        v_shard,
        mesh=mesh_tp,
        shapes={"tp": {"q": query_shapes, "k": key_shapes}},
    )

    attn_ref = attn_full.detach().clone().requires_grad_(True)
    v_ref = v_full.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("bhqk,bhkd->bhqd", attn_ref, v_ref)
    expected = torch.split(expected_full, query_shapes, dim=2)[rank]
    assert_close(z, expected)

    grad_full = torch.randn(batch, heads, query, dim)
    grad_shard = torch.split(grad_full, query_shapes, dim=2)[rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(attn_shard.grad, torch.split(attn_ref.grad, query_shapes, dim=2)[rank])
    assert_close(v_shard.grad, torch.split(v_ref.grad, key_shapes, dim=2)[rank])


def test_shared_sharded_batch_axis(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    dp_size = dist.get_world_size(dp_group)
    batch = dp_size * 2

    x = torch.randn(batch, 3, 5)
    y = torch.randn(batch, 5, 7)
    x_shard = torch.split(x, batch // dp_size, dim=0)[dp_rank].contiguous()
    y_shard = torch.split(y, batch // dp_size, dim=0)[dp_rank].contiguous()

    z = es.einshard("b/dp a c, b/dp c d -> b/dp a d", x_shard, y_shard, mesh=mesh_2d)

    expected = torch.split(torch.einsum("bac,bcd->bad", x, y), batch // dp_size, dim=0)[dp_rank]
    assert_close(z, expected)


def test_shared_batch_axis_splits_replicated_operand(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    dp_size = dist.get_world_size(dp_group)
    batch = dp_size * 2 + 1
    batch_shapes = es.helpers.compute_split_shapes(batch, dp_size)

    x_full = torch.randn(batch, 3, 5)
    y = torch.randn(batch, 5, 7, requires_grad=True)
    x_shard = torch.split(x_full, batch_shapes, dim=0)[dp_rank].clone().requires_grad_(True)

    z = es.einshard(
        "b/dp a c, b c d -> b/dp a d",
        x_shard,
        y,
        mesh=mesh_2d,
        shapes={"dp": {"b": batch_shapes}},
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    expected_full = torch.einsum("bac,bcd->bad", x_ref, y_ref)
    expected = torch.split(expected_full, batch_shapes, dim=0)[dp_rank]
    assert_close(z, expected)

    grad_full = torch.randn(batch, 3, 7)
    grad_shard = torch.split(grad_full, batch_shapes, dim=0)[dp_rank]
    z.backward(grad_shard)
    expected_full.backward(grad_full)

    assert_close(x_shard.grad, torch.split(x_ref.grad, batch_shapes, dim=0)[dp_rank])
    assert_close(y.grad, y_ref.grad)


def test_shared_sharded_batch_axis_with_sharded_contraction(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    sp_group = mesh_2d["sp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    sp_size = dist.get_world_size(sp_group)
    batch = dp_size * 2
    channels = sp_size * 3

    x = torch.randn(batch, 4, channels)
    y = torch.randn(batch, channels, 6)
    x_shard = torch.split(torch.split(x, batch // dp_size, dim=0)[dp_rank], channels // sp_size, dim=2)[sp_rank]
    y_shard = torch.split(torch.split(y, batch // dp_size, dim=0)[dp_rank], channels // sp_size, dim=1)[sp_rank]

    z = es.einshard("b/dp a c/sp, b/dp c/sp d -> b/dp a d", x_shard, y_shard, mesh=mesh_2d)

    expected = torch.split(torch.einsum("bac,bcd->bad", x, y), batch // dp_size, dim=0)[dp_rank]
    assert_close(z, expected)
