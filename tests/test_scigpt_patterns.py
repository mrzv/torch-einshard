import torch
import torch.distributed as dist
import torch.nn.functional as F

import torch_einshard as es

from conftest import assert_close


def test_tensor_parallel_mlp_pattern(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    batch, seq, channels, out_channels = 2, 3, 4, 5
    hidden = size * 3
    hidden_shapes = es.helpers.compute_split_shapes(hidden, size)

    x = torch.randn(batch, seq, channels)
    w1 = torch.randn(hidden, channels)
    b1 = torch.randn(hidden)
    w2 = torch.randn(out_channels, hidden)
    b2 = torch.randn(out_channels)

    w1_shard = torch.split(w1, hidden_shapes, dim=0)[rank].contiguous()
    b1_shard = torch.split(b1, hidden_shapes, dim=0)[rank].contiguous()
    w2_shard = torch.split(w2, hidden_shapes, dim=1)[rank].contiguous()

    h = es.einshard("b n c, h/tp c -> b n h/tp", x, w1_shard, mesh=mesh_tp)
    h = F.gelu(h + b1_shard)
    y_partial = es.einshard("b n h/tp, o h/tp -> b n o // tp", h, w2_shard, mesh=mesh_tp)
    y = es.einshard("b n o // tp -> b n o", y_partial, mesh=mesh_tp) + b2

    expected = F.gelu(torch.einsum("bnc,hc->bnh", x, w1) + b1)
    expected = torch.einsum("bnh,oh->bno", expected, w2) + b2
    assert_close(y, expected)


def test_tensor_parallel_mlp_pattern_with_ellipsis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    batch, seq, channels, out_channels = 2, 3, 4, 5
    hidden = size * 3
    hidden_shapes = es.helpers.compute_split_shapes(hidden, size)

    x = torch.randn(batch, seq, channels)
    w1 = torch.randn(hidden, channels)
    w2 = torch.randn(out_channels, hidden)
    w1_shard = torch.split(w1, hidden_shapes, dim=0)[rank].contiguous()
    w2_shard = torch.split(w2, hidden_shapes, dim=1)[rank].contiguous()

    h = es.einshard("... c, h/tp c -> ... h/tp", x, w1_shard, mesh=mesh_tp)
    y_partial = es.einshard("... h/tp, o h/tp -> ... o // tp", h, w2_shard, mesh=mesh_tp)
    y = es.einshard("... o // tp -> ... o", y_partial, mesh=mesh_tp)

    expected = torch.einsum("...c,hc->...h", x, w1)
    expected = torch.einsum("...h,oh->...o", expected, w2)
    assert_close(y, expected)


def test_tensor_parallel_attention_projection_pattern(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    batch, seq, channels, heads_per_rank, head_dim = 2, 5, 4, 2, 3
    heads = heads_per_rank * size
    embed = heads * head_dim
    embed_shapes = [heads_per_rank * head_dim for _ in range(size)]

    x = torch.randn(batch, seq, channels)
    wq = torch.randn(embed, channels)
    wk = torch.randn(embed, channels)
    wv = torch.randn(embed, channels)
    wo = torch.randn(channels, embed)

    wq_shard = torch.split(wq, embed_shapes, dim=0)[rank].contiguous()
    wk_shard = torch.split(wk, embed_shapes, dim=0)[rank].contiguous()
    wv_shard = torch.split(wv, embed_shapes, dim=0)[rank].contiguous()
    wo_shard = torch.split(wo, embed_shapes, dim=1)[rank].contiguous()

    q = es.einshard("b l c, e/tp c -> b l e/tp", x, wq_shard, mesh=mesh_tp)
    k = es.einshard("b l c, e/tp c -> b l e/tp", x, wk_shard, mesh=mesh_tp)
    v = es.einshard("b l c, e/tp c -> b l e/tp", x, wv_shard, mesh=mesh_tp)
    q = q.reshape(batch, seq, heads_per_rank, head_dim).transpose(1, 2)
    k = k.reshape(batch, seq, heads_per_rank, head_dim).transpose(1, 2)
    v = v.reshape(batch, seq, heads_per_rank, head_dim).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).reshape(batch, seq, embed_shapes[rank])
    y_partial = es.einshard("b l e/tp, c e/tp -> b l c // tp", attn, wo_shard, mesh=mesh_tp)
    y = es.einshard("b l c // tp -> b l c", y_partial, mesh=mesh_tp)

    q_full = torch.einsum("blc,ec->ble", x, wq).reshape(batch, seq, heads, head_dim).transpose(1, 2)
    k_full = torch.einsum("blc,ec->ble", x, wk).reshape(batch, seq, heads, head_dim).transpose(1, 2)
    v_full = torch.einsum("blc,ec->ble", x, wv).reshape(batch, seq, heads, head_dim).transpose(1, 2)
    expected = F.scaled_dot_product_attention(q_full, k_full, v_full)
    expected = expected.transpose(1, 2).reshape(batch, seq, embed)
    expected = torch.einsum("ble,ce->blc", expected, wo)
    assert_close(y, expected)
