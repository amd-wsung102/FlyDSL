# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""PA Decode FP8 sliding window — correctness test against torch reference."""
import sys, os, torch, math, random, gc
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

sys.path.insert(0, 'build-fly/python_packages'); sys.path.insert(1, '.')
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'

aiter = pytest.importorskip("aiter", reason="aiter is not installed, skipping PA tests")

from tests.test_common import checkAllclose
from kernels.pa_decode_fp8 import (
    build_pa_decode_module, BLOCK_THREADS, QUERY_GROUP_SIZE, HEAD_SIZE, KV_COMPUTE_BLOCK,
)
import kernels.pa_decode_fp8 as _pa
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir as _ir
import flydsl.compiler as flyc, flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T
from aiter import per_tensor_quant, dtypes as aiter_dtypes

QG = QUERY_GROUP_SIZE
fp8 = torch.float8_e4m3fnuz
bf16 = torch.bfloat16
dev = 'cuda'
SEED = 42


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_kv_caches(num_blocks, block_size, num_kv_heads, head_size):
    x = 16
    key_shape = (num_blocks, num_kv_heads, head_size // x, block_size, x)
    val_shape = (num_blocks, num_kv_heads, head_size, block_size)
    kc = torch.empty(key_shape, dtype=bf16, device=dev)
    vc = torch.empty(val_shape, dtype=bf16, device=dev)
    kc.uniform_(-1, 1)
    vc.uniform_(-1, 1)
    return kc, vc


def quantize_kv_per_tensor(key_cache, value_cache):
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    x = 16
    kc_reshaped = (key_cache.permute(0, 1, 3, 2, 4)
                   .reshape(num_blocks, num_heads, block_size, -1).contiguous())
    kc_reshaped = (kc_reshaped.view(num_blocks, num_heads, block_size,
                                     head_dim // x, x)
                   .permute(0, 1, 3, 2, 4).contiguous())
    q_keys, key_scale = per_tensor_quant(kc_reshaped, quant_dtype=aiter_dtypes.fp8)
    q_vals, val_scale = per_tensor_quant(value_cache, quant_dtype=aiter_dtypes.fp8)
    return q_keys, key_scale, q_vals, val_scale


def torch_ref_attention_sw(query, key_cache, value_cache, block_tables,
                           context_lengths, key_scale, value_scale,
                           sliding_window):
    """Reference attention restricted to the last ``sliding_window`` tokens."""
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    softmax_scale = 1.0 / math.sqrt(head_dim)
    batch_size = query.shape[0]
    num_query_heads = query.shape[1]
    kc_flat = (key_cache.permute(0, 3, 1, 2, 4)
               .contiguous().view(-1, num_heads, head_dim))
    vc_flat = (value_cache.permute(0, 3, 1, 2)
               .contiguous().view(-1, num_heads, head_dim))
    kv_dtype = key_cache.dtype
    outputs = []
    for b in range(batch_size):
        bt = block_tables[b]
        ctx_len = context_lengths[b].item()
        window_start = max(0, ctx_len - sliding_window)

        all_tok_idx = (bt.repeat_interleave(block_size)[:ctx_len] * block_size
                       + torch.arange(ctx_len, device=dev) % block_size)
        tok_idx = all_tok_idx[window_start:]

        keys = kc_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        if key_scale is not None:
            keys = keys * key_scale.item()
        vals = vc_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        if value_scale is not None:
            vals = vals * value_scale.item()
        q_b = query[b].float()
        qg = num_query_heads // num_heads
        q_grouped = q_b.view(num_heads, qg, head_dim)
        scores = torch.einsum('gqd,tgd->gqt', q_grouped, keys) * softmax_scale
        attn = torch.softmax(scores, dim=-1)
        out_b = torch.einsum('gqt,tgd->gqd', attn, vals)
        outputs.append(out_b.reshape(num_query_heads, head_dim))
    return torch.stack(outputs).to(query.dtype)


def run_sliding_window_test(num_query_heads, num_kv_heads, batch_size,
                            context_length, sliding_window_size,
                            block_size=16):
    """Build the FlyDSL PA kernel with sliding_window and compare to a
    torch reference that only attends to the last ``sliding_window_size``
    tokens."""
    setup_seed(SEED)
    head_size = HEAD_SIZE
    softmax_scale = 1.0 / math.sqrt(head_size)
    qg = num_query_heads // num_kv_heads
    assert qg == QG

    max_blocks_per_seq = (context_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_seq * batch_size
    blocks_per_seq = max_blocks_per_seq

    effective_len = min(context_length, sliding_window_size)
    num_parts = (effective_len + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
    _one_shot = num_parts <= 1

    query = torch.empty(batch_size, num_query_heads, head_size,
                        dtype=bf16, device=dev)
    query.uniform_(-1, 1)
    kc_bf16, vc_bf16 = create_kv_caches(
        total_blocks, block_size, num_kv_heads, head_size)
    q_keys, key_scale, q_vals, val_scale = quantize_kv_per_tensor(
        kc_bf16, vc_bf16)

    block_tables = torch.tensor(
        [[random.randint(0, total_blocks - 1) for _ in range(blocks_per_seq)]
         for _ in range(batch_size)],
        dtype=torch.int32, device=dev)
    context_lengths = torch.full(
        (batch_size,), context_length, dtype=torch.int32, device=dev)

    fd_query, fd_q_scale = per_tensor_quant(query, quant_dtype=aiter_dtypes.fp8)
    _qs = fd_q_scale.item()

    ref_out = torch_ref_attention_sw(
        query, q_keys, q_vals, block_tables, context_lengths,
        key_scale, val_scale, sliding_window_size)

    # ── FlyDSL kernel with sliding window ────────────────────────
    fd_kfn = build_pa_decode_module(
        batch_size, num_kv_heads, num_parts,
        blocks_per_seq, kv_block_size=block_size,
        softmax_scale=softmax_scale,
        query_scale=_qs,
        key_scale=key_scale.item(),
        value_scale=val_scale.item(),
        one_shot=_one_shot,
        sliding_window=sliding_window_size)
    fd_al = _pa.allocator

    if _one_shot:
        fd_out = torch.zeros(batch_size, num_kv_heads, 1, qg, head_size,
                             dtype=bf16, device=dev)
        fd_es = torch.zeros(1, dtype=torch.float32, device=dev)
        fd_ml = torch.zeros(1, dtype=torch.float32, device=dev)
    else:
        fd_out = torch.zeros(batch_size, num_kv_heads, num_parts, qg,
                             head_size, dtype=bf16, device=dev)
        fd_es = torch.zeros(batch_size, num_kv_heads, num_parts, qg,
                             dtype=torch.float32, device=dev)
        fd_ml = torch.full((batch_size, num_kv_heads, num_parts, qg),
                           float('-inf'), dtype=torch.float32, device=dev)

    grid_z = 1 if _one_shot else num_parts
    _cache_tag = (batch_size, num_kv_heads, grid_z, sliding_window_size)

    @flyc.jit
    def fd_launch(out, es, ml, q, kc, vc, bt, cl: fx.Int32,
                  gx: fx.Int32, gy: fx.Int32, gz: fx.Int32,
                  stream: fx.Stream):
        _ = _cache_tag
        fd_al.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            fd_al.finalize()
        grid_x = arith.index_cast(T.index, gx.ir_value())
        grid_y = arith.index_cast(T.index, gy.ir_value())
        grid_z_val = arith.index_cast(T.index, gz.ir_value())
        fd_kfn(out, es, ml, q, kc, vc, bt, cl).launch(
            grid=(grid_x, grid_y, grid_z_val),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    fd_launch(fd_out, fd_es, fd_ml, fd_query, q_keys, q_vals,
              block_tables, context_length,
              batch_size, num_kv_heads, grid_z,
              torch.cuda.current_stream())
    torch.cuda.synchronize()

    if _one_shot:
        fd_final = fd_out.squeeze(2)
    else:
        from aiter.ops.triton.gluon.pa_decode_gluon import (
            _paged_attention_decode_v2_reduce_kernel_wrapper,
        )
        reduce_out = torch.empty(batch_size, 1, num_kv_heads, qg, head_size,
                                 dtype=bf16, device=dev)
        _paged_attention_decode_v2_reduce_kernel_wrapper(
            (batch_size, num_kv_heads, 1),
            reduce_out, fd_es, fd_ml, fd_out, context_lengths, None,
            reduce_out.stride(0), reduce_out.stride(1),
            reduce_out.stride(2), reduce_out.stride(3),
            fd_es.stride(0), fd_es.stride(1), fd_es.stride(2),
            fd_out.stride(0), fd_out.stride(1), fd_out.stride(2),
            fd_out.stride(3),
            query_seq_len=1, query_group_size=qg,
            HEAD_SIZE=head_size, CONTEXT_PARTITION_SIZE=KV_COMPUTE_BLOCK,
            PS=False, context_partition_num=num_parts,
        )
        torch.cuda.synchronize()
        fd_final = reduce_out.reshape(batch_size, num_kv_heads, qg, head_size)

    # ── Verify ───────────────────────────────────────────────────
    fd_flat = fd_final.reshape(batch_size, num_query_heads, head_size).float()
    ref_flat = ref_out.float()

    diff_tol = 5e-2
    err = checkAllclose(ref_flat, fd_flat, atol=diff_tol, rtol=diff_tol,
                        msg=f"[SW ref vs FlyDSL] ctx={context_length} sw={sliding_window_size}")
    torch.cuda.empty_cache()
    gc.collect()
    return err


@pytest.mark.parametrize("ctx,sw", [
    (512, 256),
    (1024, 256),
    (1024, 512),
])
def test_pa_decode_sliding_window(ctx, sw):
    err = run_sliding_window_test(
        num_query_heads=16, num_kv_heads=1,
        batch_size=4, context_length=ctx,
        sliding_window_size=sw, block_size=16)
    assert err < 0.05, f"Sliding window error too high: {err:.4f}"


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    configs = [
        (512, 256),
        (1024, 256),
        (1024, 512),
        (2048, 512),
    ]
    print(f"{'ctx':>6} {'sw':>6} | {'error':>8} | status")
    print("-" * 40)
    for ctx, sw in configs:
        try:
            err = run_sliding_window_test(
                num_query_heads=16, num_kv_heads=1,
                batch_size=4, context_length=ctx,
                sliding_window_size=sw, block_size=16)
            status = "PASS" if err < 0.05 else "FAIL"
            print(f"{ctx:>6} {sw:>6} | {err:>8.4f} | {status}")
        except Exception:
            import traceback; traceback.print_exc()
            print(f"{ctx:>6} {sw:>6} | {'N/A':>8} | ERROR")
