# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE token sorting kernel (FlyDSL).

Implements the MoE sorting operation used in DeepSeek R1 and similar MoE models.
Given router top-k selections (topk_ids, topk_weights), reorganizes tokens by expert
for efficient batched expert GEMM execution.

Algorithm: counting sort in LDS (histogram → prefix-sum → scatter).

Two paths:
  - Decode (small T): single kernel, all phases in LDS.
  - Prefill (large T): 4 kernels via HBM workspace (ClearWS → P0 scatter → P1 count → P23 prefix-sum+scatter).

Packed token ID format: (topk_position << 24) | token_id
  - Upper 8 bits: topk slot (0..topk-1)
  - Lower 24 bits: token index (0..M-1)
  - Padding sentinel: (topk << 24) | M
"""

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, memref as memref_ops
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, buffer_ops, const_expr, range_constexpr, vector as fly_vector
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.kernels_common import get_warp_size

BLOCK_SIZE = 256
UNIT_SIZE = 32  # GEMM tile-M, aka block_size in CK

# ---------------------------------------------------------------------------
# AOT-compiled dispatch caches — keyed by constexpr values.
# After the first JIT call (which compiles the kernel), flyc.compile()
# returns a CompiledFunction whose __call__ skips inspect.Signature.bind,
# _make_cache_key, and dict lookup, reducing dispatch from ~70 us to ~5 us.
# ---------------------------------------------------------------------------
_decode_cf_cache = {}   # (num_experts, topk, max_tokens, unit_size, n_grid_blocks) -> CompiledFunction
_prefill_cf_cache = {}  # (num_experts, topk, unit_size, kernel_name, *constexpr_vals) -> CompiledFunction


@contextmanager
def _if_then(if_op):
    """Context manager for scf.IfOp then-region (from moe_gemm_2stage.py)."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


# ---------------------------------------------------------------------------
# FlyDSL GPU kernel — decode path (single kernel, SubTokenOneShot)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def compile_moe_sorting_decode(
    *,
    num_experts: int,
    topk: int,
    max_tokens: int = 128,
    unit_size: int = UNIT_SIZE,
):
    """Compile the decode-path MoE sorting kernel.

    Parameters
    ----------
    num_experts : int
        Number of routed experts (e.g. 256 for DeepSeek R1).
    topk : int
        Experts per token (e.g. 8 for DeepSeek R1).
    max_tokens : int
        Upper bound on T for LDS sizing. Actual T is passed at runtime.
    unit_size : int
        GEMM tile-M for padding alignment (default 32).
    """
    arch = get_hip_arch()
    WARP_SIZE = get_warp_size(arch)
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE
    E = num_experts
    smem_cols = E + 1

    # LDS sizing: sub_tokens rows for the token×expert histogram
    # Match CK's sizing: total LDS / occupancy / smem_cols, rounded to 8
    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536  # conservative default

    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        raise ValueError(
            f"LDS too small for E={E}: need at least "
            f"{(cumsum_bufs + sub_unroll) * smem_cols * 4} bytes"
        )
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    r_token_min = ((max_tokens + sub_unroll - 1) // sub_unroll) * sub_unroll
    r_for_sub = min(r_for_sub, r_token_min)
    sub_tokens = r_for_sub
    smem_rows = sub_tokens + cumsum_bufs  # 2 extra rows for cumsum/cumdup

    total_smem_ints = smem_rows * smem_cols
    total_smem_bytes = total_smem_ints * 4

    # SmemAllocator for the 3 LDS regions
    allocator = SmemAllocator(None, arch=arch)

    # Region 0: cumsum[E+1]  (exclusive prefix sums per expert)
    cumsum_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = cumsum_offset + smem_cols * 4

    # Region 1: cumdup[E+1]  (duplicate of cumsum for scatter phase)
    cumdup_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = cumdup_offset + smem_cols * 4

    # Region 2: tokens_mesh[sub_tokens, smem_cols]
    mesh_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = mesh_offset + sub_tokens * smem_cols * 4

    # Helpers for raw memref LDS access (used inside scf.for instead of SmemPtr)
    def _to_index(v):
        """Convert i32 or index DSL value to raw MLIR index value."""
        raw = v.ir_value() if hasattr(v, 'ir_value') else v
        if isinstance(raw.type, ir.IndexType):
            return raw
        return ArithValue(v).index_cast(T.index)

    def _lds_load(raw_mr, idx):
        """Load i32 from LDS raw memref. idx can be i32 or index."""
        return fx.Int32(memref_ops.load(raw_mr, [_to_index(idx)]))

    def _lds_store(raw_mr, val, idx):
        """Store i32 to LDS raw memref. idx can be i32 or index."""
        v = val.ir_value() if hasattr(val, 'ir_value') else val
        memref_ops.store(v, raw_mr, [_to_index(idx)])

    def _unwrap(v):
        """Unwrap DSL value to raw MLIR ir.Value for scf.for init."""
        return v.ir_value() if hasattr(v, 'ir_value') else v

    @flyc.kernel
    def moe_sorting_decode_kernel(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        tokens = i32_tokens
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        c4_i32 = fx.Int32(4)

        # Buffer resources (needed by both paths, defined at top level)
        moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids_tensor, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(topk_weights_tensor, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights_out, max_size=True)
        sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        # LDS: get RAW memrefs ONCE — dominates all child scf.for/scf.if regions.
        base_ptr = allocator.get_base()
        cumsum_mr = SmemPtr(base_ptr, cumsum_offset, T.i32, shape=(smem_cols,)).get()
        cumdup_mr = SmemPtr(base_ptr, cumdup_offset, T.i32, shape=(smem_cols,)).get()
        mesh_mr = SmemPtr(base_ptr, mesh_offset, T.i32, shape=(sub_tokens * smem_cols,)).get()

        c_topk = fx.Int32(topk)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_sub_tokens = fx.Int32(sub_tokens)
        c_smem_cols = fx.Int32(smem_cols)
        c_sentinel = fx.Int32((topk << 24))

        # =================== MOE_BUF ZEROING (blocks > 0 only) ===============
        is_zero_block = arith.cmpi(arith.CmpIPredicate.ne, bid, c_zero_i32)
        _if_zero = scf.IfOp(is_zero_block)
        with _if_then(_if_zero):
            zero_gid = (bid - c_one_i32) * fx.Int32(BLOCK_SIZE) + tid
            num_zero_blocks = fx.grid_dim.x - c_one_i32
            zero_stride = num_zero_blocks * fx.Int32(BLOCK_SIZE)
            # Opt 10: compute loop bound from moe_buf_elems / stride (avoid 255 wasted iterations)
            zero_niters = (i32_moe_buf_elems + zero_stride - c_one_i32) // zero_stride
            _zs = arith.index(0)
            _ze = ArithValue(zero_niters).index_cast(T.index)
            _z1 = arith.index(1)
            for _z in range(_zs, _ze, _z1):
                z_idx = zero_gid + fx.Int32(_z) * zero_stride
                z_valid = z_idx < i32_moe_buf_elems
                z_safe = z_valid.select(z_idx, c_oob_idx)
                buffer_ops.buffer_store(c_zero_i32, moe_buf_rsrc, z_safe)

        # =================== SORTING (block 0 only) ==========================
        is_sort_block = arith.cmpi(arith.CmpIPredicate.eq, bid, c_zero_i32)
        _if_sort = scf.IfOp(is_sort_block)
        with _if_then(_if_sort):
            # ========================= PHASE 1: Histogram =========================
            # Clear mesh region — unconditional store to safe index when out of bounds
            for i_clear in range_constexpr(0, sub_tokens * smem_cols, BLOCK_SIZE):
                idx = fx.Int32(i_clear) + tid
                is_valid = idx < fx.Int32(sub_tokens * smem_cols)
                safe_idx = is_valid.select(idx, c_zero_i32)
                safe_idx_ix = ArithValue(safe_idx).index_cast(T.index)
                # Always store; out-of-bounds threads harmlessly write to index 0
                _lds_store(mesh_mr, c_zero_i32, safe_idx_ix)
            gpu.barrier()

            # Fill mesh: for each (token, topk_slot), write topk_slot+1 to mesh[token, expert_id]
            total_assignments = tokens * c_topk
            for i_assign in range_constexpr(0, max_tokens * topk, BLOCK_SIZE):
                flat_idx = fx.Int32(i_assign) + tid
                is_valid = flat_idx < total_assignments
                safe_flat = is_valid.select(flat_idx, c_zero_i32)

                token_id = safe_flat // c_topk
                topk_slot = safe_flat % c_topk

                global_idx = token_id * c_topk + topk_slot
                eid = buffer_ops.buffer_load(topk_ids_rsrc, global_idx, vec_width=1, dtype=T.i32)

                # mesh[token_id, eid] = topk_slot + 1; invalid threads write 0 to index 0
                mesh_addr = token_id * c_smem_cols + eid
                safe_mesh_addr = is_valid.select(mesh_addr, c_zero_i32)
                safe_mesh_ix = ArithValue(safe_mesh_addr).index_cast(T.index)
                val = is_valid.select(topk_slot + c_one_i32, c_zero_i32)
                _lds_store(mesh_mr, val, safe_mesh_ix)
            gpu.barrier()

            # ===================== PHASE 2: Count + Prefix Sum =====================
            c_lane_group_sz = fx.Int32(8)
            lane_group_id = tid // c_lane_group_sz
            lane_group_os = tid % c_lane_group_sz
            width8_i32 = fx.Int32(8)
            width_ws = fx.Int32(WARP_SIZE)

            is_t0 = (tid == c_zero_i32)

            # Initialize cumsum[0] = 0.  All threads write 0 so there's no
            # read-modify-write race across waves.
            _lds_store(cumsum_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            for i_e in range_constexpr(0, E, BLOCK_SIZE // 8):
                eid_local = fx.Int32(i_e) + lane_group_id
                eid_valid = eid_local < c_E

                cnt = c_zero_i32
                for i_sub in range_constexpr(0, sub_tokens, 8):
                    sub_idx = fx.Int32(i_sub) + lane_group_os
                    sub_valid = sub_idx < c_sub_tokens
                    combined_valid = eid_valid & sub_valid

                    safe_sub = combined_valid.select(sub_idx, c_zero_i32)
                    safe_eid = combined_valid.select(eid_local, c_zero_i32)
                    mesh_rd_addr = safe_sub * c_smem_cols + safe_eid
                    mesh_rd_ix = ArithValue(mesh_rd_addr).index_cast(T.index)
                    mesh_val = _lds_load(mesh_mr, mesh_rd_ix)

                    has_token = combined_valid.select(
                        arith.cmpi(arith.CmpIPredicate.ne, mesh_val, c_zero_i32).select(c_one_i32, c_zero_i32),
                        c_zero_i32,
                    )

                    # Reduce within lane-group of 8
                    reduced = has_token
                    for sh in range_constexpr(3):
                        off = fx.Int32(1 << sh)
                        peer = reduced.shuffle_xor(off, width8_i32)
                        reduced = reduced + peer
                    cnt = cnt + reduced

                # Only lane 0 of each valid lane-group writes the count to cumsum[eid+1].
                # Invalid threads: write_valid is false, cs_idx = 0, and we write 0 to
                # cumsum[0] which is harmless (cumsum[0] is always 0).
                write_valid = eid_valid & (lane_group_os == c_zero_i32)
                cs_idx = write_valid.select(eid_local + c_one_i32, c_zero_i32)
                cs_ix = ArithValue(cs_idx).index_cast(T.index)
                cs_val = write_valid.select(cnt, c_zero_i32)
                _lds_store(cumsum_mr, cs_val, cs_ix)
            gpu.barrier()

            # Phase 2b: Prefix sum over expert counts.
            # Step 1: Each thread converts its expert's raw count → padded block size.
            for i_cvt in range_constexpr(0, E, BLOCK_SIZE):
                cvt_eid = fx.Int32(i_cvt) + tid
                cvt_valid = cvt_eid < c_E
                # Safe index: valid → cumsum[eid+1], invalid → cumsum[0] (write 0, harmless)
                safe_cvt_idx = cvt_valid.select(cvt_eid + c_one_i32, c_zero_i32)
                cvt_ix = ArithValue(safe_cvt_idx).index_cast(T.index)
                raw_cnt_cvt = _lds_load(cumsum_mr, cvt_ix)
                blocks_cvt = (raw_cnt_cvt + c_unit - c_one_i32) // c_unit
                padded_cvt = arith.cmpi(arith.CmpIPredicate.eq, raw_cnt_cvt, c_zero_i32).select(
                    c_zero_i32, blocks_cvt * c_unit)
                # Valid threads write padded value; invalid threads write 0 to cumsum[0]
                _lds_store(cumsum_mr, cvt_valid.select(padded_cvt, c_zero_i32), cvt_ix)
            gpu.barrier()

            # Step 2: Parallel prefix sum using DPP row_shr + ds_bpermute.
            # Wave 0 does the prefix sum; other waves idle.
            # For E <= WARP_SIZE (e.g. E=32): single pass.
            # For E > WARP_SIZE (e.g. E=256): process in chunks of WARP_SIZE.
            # DPP row_shr (1 cycle, register-to-register) replaces ds_bpermute
            # (2-4 cycles, goes through LDS) for intra-row shifts (1,2,4,8).
            # Cross-row shifts (16,32) still use ds_bpermute.
            is_wave0 = wave == c_zero_i32
            prev_chunk_total = c_zero_i32

            # DPP constants
            DPP_ROW_SHR_1 = 0x111
            DPP_ROW_SHR_2 = 0x112
            DPP_ROW_SHR_4 = 0x114
            DPP_ROW_SHR_8 = 0x118
            DPP_ROW_MASK = 0xF
            DPP_BANK_MASK = 0xF

            for chunk_start in range_constexpr(0, E, WARP_SIZE):
                # Each lane handles one expert in this chunk
                eid_ps = fx.Int32(chunk_start) + lane
                eid_ps_valid = is_wave0 & (eid_ps < c_E)
                safe_eid_ps = eid_ps_valid.select(eid_ps + c_one_i32, c_zero_i32)
                ps_ix = ArithValue(safe_eid_ps).index_cast(T.index)
                val = eid_ps_valid.select(_lds_load(cumsum_mr, ps_ix), c_zero_i32)

                # Hillis-Steele inclusive prefix sum:
                # Steps 0-3 (shift 1,2,4,8) use DPP row_shr (intra-row, 1 cycle)
                # Steps 4-5 (shift 16,32) use ds_bpermute (cross-row)
                val_raw = _unwrap(val)
                zero_raw = _unwrap(c_zero_i32)

                # DPP row_shr:1 — shift right by 1 within 16-lane row
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                              DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True)
                should_add = lane >= c_one_i32
                val = should_add.select(val + fx.Int32(remote), val)

                # DPP row_shr:2
                val_raw = _unwrap(val)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                              DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True)
                should_add = lane >= fx.Int32(2)
                val = should_add.select(val + fx.Int32(remote), val)

                # DPP row_shr:4
                val_raw = _unwrap(val)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                              DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True)
                should_add = lane >= fx.Int32(4)
                val = should_add.select(val + fx.Int32(remote), val)

                # DPP row_shr:8
                val_raw = _unwrap(val)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                              DPP_ROW_SHR_8, DPP_ROW_MASK, DPP_BANK_MASK, True)
                should_add = lane >= fx.Int32(8)
                val = should_add.select(val + fx.Int32(remote), val)

                if WARP_SIZE > 16:
                    # Cross-row: shift by 16 via ds_bpermute
                    # Read from last lane of previous row: (lane & 0x30) - 1
                    src_lane_16 = (lane & fx.Int32(0x30)) - c_one_i32
                    src_addr_16 = src_lane_16 * c4_i32
                    remote16 = fly_rocdl.ds_bpermute(T.i32, src_addr_16, val)
                    should_add16 = lane >= fx.Int32(16)
                    val = should_add16.select(val + fx.Int32(remote16), val)

                if WARP_SIZE > 32:
                    # Cross-row: shift by 32 via ds_bpermute
                    src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
                    src_addr_32 = src_lane_32 * c4_i32
                    remote32 = fly_rocdl.ds_bpermute(T.i32, src_addr_32, val)
                    should_add32 = lane >= fx.Int32(32)
                    val = should_add32.select(val + fx.Int32(remote32), val)

                # Add previous chunk's total
                val = val + prev_chunk_total

                # Write result to cumdup (avoid racing on cumsum)
                _lds_store(cumdup_mr, eid_ps_valid.select(val, c_zero_i32),
                           eid_ps_valid.select(eid_ps + c_one_i32, c_zero_i32))

                # Get last lane's value as prev_chunk_total for next chunk
                last_addr = fx.Int32((WARP_SIZE - 1) * 4)
                prev_chunk_total = fly_rocdl.ds_bpermute(T.i32, last_addr, val)
                prev_chunk_total = fx.Int32(prev_chunk_total)

            # cumdup[0] = 0
            _lds_store(cumdup_mr, is_t0.select(c_zero_i32, _lds_load(cumdup_mr, c_zero_i32)), c_zero_i32)
            gpu.barrier()

            # Write num_valid_ids from cumdup[E]
            cs_E_ix_ps = ArithValue(c_E).index_cast(T.index)
            total_padded = _lds_load(cumdup_mr, cs_E_ix_ps)
            buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero_i32)
            buffer_ops.buffer_store(tokens, nvalid_rsrc, c_one_i32)
            gpu.barrier()

            # Copy cumdup → cumsum (all threads, one expert per thread)
            for i_cp in range_constexpr(0, E + 1, BLOCK_SIZE):
                cp_idx = fx.Int32(i_cp) + tid
                cp_valid = cp_idx <= c_E
                safe_cp_idx = cp_valid.select(cp_idx, c_zero_i32)
                cp_ix = ArithValue(safe_cp_idx).index_cast(T.index)
                cp_val = _lds_load(cumdup_mr, cp_ix)
                _lds_store(cumsum_mr, cp_val, cp_ix)
            gpu.barrier()

            # Write sorted_expert_ids — predicated stores to buffer (safe: buffer_store ignores OOB)
            for i_eid in range_constexpr(0, E, BLOCK_SIZE):
                eid_wr = fx.Int32(i_eid) + tid
                eid_wr_valid = eid_wr < c_E
                safe_eid_wr = eid_wr_valid.select(eid_wr, c_zero_i32)

                cs_start_ix = ArithValue(safe_eid_wr).index_cast(T.index)
                cs_end_ix = ArithValue(safe_eid_wr + c_one_i32).index_cast(T.index)
                e_start = _lds_load(cumsum_mr, cs_start_ix)
                e_end = eid_wr_valid.select(_lds_load(cumsum_mr, cs_end_ix), e_start)

                # Store cumdup: valid threads write e_start to cumdup[eid],
                # invalid threads write cumsum[0]=0 to cumdup[0] (harmless).
                _lds_store(cumdup_mr, e_start, cs_start_ix)

                blk_start = e_start // c_unit
                blk_end = e_end // c_unit
                for j_blk in range_constexpr(max_tokens):
                    blk_idx = blk_start + fx.Int32(j_blk)
                    blk_valid = eid_wr_valid & (blk_idx < blk_end)
                    # buffer_store with predicated index (OOB writes are dropped by HW)
                    safe_blk = blk_valid.select(blk_idx, c_oob_idx)
                    buffer_ops.buffer_store(eid_wr, sorted_e_rsrc, safe_blk)
            gpu.barrier()

            # Store cumdup[E] = cumsum[E].
            # All threads write cumE to cumdup[E] (all write the same value, no race).
            cs_E_ix = ArithValue(c_E).index_cast(T.index)
            cumE = _lds_load(cumsum_mr, cs_E_ix)
            _lds_store(cumdup_mr, cumE, cs_E_ix)
            gpu.barrier()

            # ====================== PHASE 3: Scatter ==============================
            # CK uses wave_cumsum<8> (DPP prefix sum) + ds_bpermute (lane broadcast).
            # FlyDSL has neither. Instead, each lane reads all 8 mesh values in the
            # batch from LDS to compute its own prefix offset. No shuffle needed.
            for i_e2 in range_constexpr(0, E, BLOCK_SIZE // 8):
                eid_sc = fx.Int32(i_e2) + lane_group_id
                eid_sc_valid = eid_sc < c_E
                # Invalid lane groups map to cumsum[E] (the total count) instead of
                # cumsum[0] to avoid racing with lane_group 0's position write-back.
                safe_eid_sc = eid_sc_valid.select(eid_sc, c_E)

                cs_sc_ix = ArithValue(safe_eid_sc).index_cast(T.index)
                position = _lds_load(cumsum_mr, cs_sc_ix)

                for i_sub2 in range_constexpr(0, sub_tokens, 8):
                    # This lane handles sub_token (i_sub2 + lane_group_os).
                    my_sub = fx.Int32(i_sub2) + lane_group_os
                    my_sub_valid = eid_sc_valid & (my_sub < c_sub_tokens)
                    safe_my_sub = my_sub_valid.select(my_sub, c_zero_i32)
                    my_mesh_addr = safe_my_sub * c_smem_cols + safe_eid_sc
                    my_mesh_ix = ArithValue(my_mesh_addr).index_cast(T.index)
                    my_x = _lds_load(mesh_mr, my_mesh_ix)
                    my_has_token = my_sub_valid & arith.cmpi(arith.CmpIPredicate.ne, my_x, c_zero_i32)
                    local_cnt = my_has_token.select(c_one_i32, c_zero_i32)

                    # Opt 7: DPP row_shr inclusive prefix sum within 8-lane group
                    # (1-cycle register-to-register vs 2-4 cycle ds_bpermute via LDS)
                    # row_shr operates within 16-lane rows; lane_group_os guard
                    # handles 8-lane group boundaries (discards cross-group values).
                    cnt_raw = _unwrap(local_cnt)
                    zero_raw = _unwrap(c_zero_i32)

                    # row_shr:1
                    remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                                  DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True)
                    should_add = lane_group_os >= c_one_i32
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    # row_shr:2
                    cnt_raw = _unwrap(local_cnt)
                    remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                                  DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True)
                    should_add = lane_group_os >= fx.Int32(2)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    # row_shr:4
                    cnt_raw = _unwrap(local_cnt)
                    remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                                  DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True)
                    should_add = lane_group_os >= fx.Int32(4)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    # Broadcast batch total from last lane of group via ds_bpermute
                    last_lane_of_group = (tid | fx.Int32(7))  # tid with lower 3 bits set
                    last_addr = last_lane_of_group * c4_i32
                    batch_total = fly_rocdl.ds_bpermute(T.i32, last_addr, local_cnt)
                    batch_total = fx.Int32(batch_total)

                    # Scatter this lane's token
                    slot = position + local_cnt - c_one_i32
                    safe_x = my_has_token.select(my_x, c_one_i32)
                    topk_slot_sc = safe_x - c_one_i32
                    packed_id = (topk_slot_sc << fx.Int32(24)) | my_sub
                    safe_slot = my_has_token.select(slot, c_oob_idx)
                    buffer_ops.buffer_store(packed_id, sorted_ids_rsrc, safe_slot)

                    w_addr = my_has_token.select(my_sub * c_topk + topk_slot_sc, c_zero_i32)
                    w_val_i32 = buffer_ops.buffer_load(weights_rsrc, w_addr, vec_width=1, dtype=T.i32)
                    buffer_ops.buffer_store(w_val_i32, sorted_w_rsrc, safe_slot)

                    # Advance position by batch total
                    position = position + batch_total

                # Write back updated position (for padding phase).
                # Invalid lane groups write position (=0+0=0) to cumsum[0] which is harmless.
                _lds_store(cumsum_mr, position, cs_sc_ix)
            gpu.barrier()

            # Fill padding slots with sentinel
            sentinel_val = c_sentinel | tokens
            # Store 0.0f as i32 bitpattern (0x00000000) to avoid f32*i32 type mismatch
            c_zero_as_i32 = c_zero_i32  # IEEE 754: 0.0f == 0x00000000
            for i_pad in range_constexpr(0, E, BLOCK_SIZE):
                eid_pad = fx.Int32(i_pad) + tid
                pad_valid = eid_pad < c_E
                safe_eid_pad = pad_valid.select(eid_pad, c_zero_i32)

                cs_pad_ix = ArithValue(safe_eid_pad).index_cast(T.index)
                cdp_ix = ArithValue(safe_eid_pad + c_one_i32).index_cast(T.index)
                pad_start = _lds_load(cumsum_mr, cs_pad_ix)
                pad_end = pad_valid.select(_lds_load(cumdup_mr, cdp_ix), pad_start)

                for j_pad in range_constexpr(unit_size):
                    pad_slot = pad_start + fx.Int32(j_pad)
                    pad_slot_valid = pad_valid & (pad_slot < pad_end)
                    safe_pad_slot = pad_slot_valid.select(pad_slot, c_oob_idx)
                    buffer_ops.buffer_store(sentinel_val, sorted_ids_rsrc, safe_pad_slot)
                    # Use sorted_ids_rsrc (i32) to store 0x00000000 (= 0.0f) into weight slot
                    # Both buffers start at the same base offset in element-width terms
                    buffer_ops.buffer_store(c_zero_as_i32, sorted_w_rsrc, safe_pad_slot)

    # end kernel

    @flyc.jit
    def launch_moe_sorting_decode(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid_blocks: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        launcher = moe_sorting_decode_kernel(
            topk_ids_tensor,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            i32_tokens,
            i32_moe_buf_elems,
        )
        # Grid: block 0 = sorting, blocks 1..N = moe_buf zeroing
        launcher.launch(
            grid=(n_grid_blocks, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_sorting_decode


# ---------------------------------------------------------------------------
# LDS helpers for prefill kernels (module-level, used inside @flyc.kernel)
# ---------------------------------------------------------------------------
def _lds_load_raw(raw_mr, idx):
    """Load i32 from LDS raw memref. idx can be i32 or index."""
    raw_idx = idx.ir_value() if hasattr(idx, 'ir_value') else idx
    if not isinstance(raw_idx.type, ir.IndexType):
        raw_idx = ArithValue(idx).index_cast(T.index)
        raw_idx = raw_idx.ir_value() if hasattr(raw_idx, 'ir_value') else raw_idx
    return fx.Int32(memref_ops.load(raw_mr, [raw_idx]))


def _lds_store_raw(raw_mr, val, idx):
    """Store i32 to LDS raw memref. idx can be i32 or index."""
    v = val.ir_value() if hasattr(val, 'ir_value') else val
    raw_idx = idx.ir_value() if hasattr(idx, 'ir_value') else idx
    if not isinstance(raw_idx.type, ir.IndexType):
        raw_idx = ArithValue(idx).index_cast(T.index)
        raw_idx = raw_idx.ir_value() if hasattr(raw_idx, 'ir_value') else raw_idx
    memref_ops.store(v, raw_mr, [raw_idx])


def _unwrap_raw(v):
    """Unwrap DSL value to raw MLIR ir.Value."""
    return v.ir_value() if hasattr(v, 'ir_value') else v


# ---------------------------------------------------------------------------
# FlyDSL GPU kernels — prefill path (4 kernels, large T via HBM workspace)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def compile_moe_sorting_prefill(
    *,
    num_experts: int,
    topk: int,
    unit_size: int = UNIT_SIZE,
):
    """Compile the prefill-path MoE sorting kernels.

    For token counts exceeding LDS capacity, uses HBM workspace:
      K1: ClearWorkspace — zero the workspace buffer
      K2: P0 scatter     — scatter topk_ids into expert mesh in HBM
      K3: P1 count       — one block per expert, count non-zero mesh cells
      K4: P23 prefix-sum + scatter — prefix-sum on counts, scatter tokens,
          fill sorted_expert_ids, zero moe_buf
      P0_v2: Fused clear+scatter+count — replaces K1+K2+K3 for small T (<=512)

    Workspace layout (i32 elements):
      [0 .. ws_mesh_i32)                : uint8 expert mesh (E rows x mesh_stride bytes, packed into i32)
      [ws_mesh_i32 .. ws_mesh_i32 + E+1): expert_cumsum (E+1 i32 entries)

    Parameters
    ----------
    num_experts : int
        Number of routed experts (e.g. 256 for DeepSeek R1).
    topk : int
        Experts per token (e.g. 8).
    unit_size : int
        GEMM tile-M for padding alignment (default 32).
    """
    arch = get_hip_arch()
    WARP_SIZE = get_warp_size(arch)
    E = num_experts

    # --- K1: ClearWorkspace kernel -------------------------------------------
    # CK uses grid=262144, block=1024 (1 store per thread, no loop).
    # Match that: block=1024, grid=ceil(ws_total/1024).
    K1_BLOCK = 1024

    @flyc.kernel(known_block_size=[K1_BLOCK, 1, 1])
    def clear_workspace_kernel(
        workspace: fx.Tensor,
        i32_total_elems: fx.Int32,
    ):
        gid = fx.block_idx.x * fx.Int32(K1_BLOCK) + fx.thread_idx.x
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)

        # Each thread stores exactly one element (no loop needed).
        valid = gid < i32_total_elems
        buffer_ops.buffer_store(c_zero, ws_rsrc, valid.select(gid, c_zero))

    @flyc.jit
    def launch_clear_ws(
        workspace: fx.Tensor,
        i32_total_elems: fx.Int32,
        n_grid: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = clear_workspace_kernel(workspace, i32_total_elems)
        launcher.launch(grid=(n_grid, 1, 1), block=(K1_BLOCK, 1, 1), stream=stream)

    # --- K2: P0 scatter kernel -----------------------------------------------
    # uint8 mesh: stores topk_slot+1 (max 9) as a single byte directly.
    # mesh_stride is in bytes; byte_offset = eid * mesh_stride + token_id.
    # No two threads write the same byte (unique experts per token).
    K2_BLOCK = 256

    @flyc.kernel
    def p0_scatter_kernel(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_niters: fx.Int32,
    ):
        gid = fx.block_idx.x * fx.Int32(K2_BLOCK) + fx.thread_idx.x
        stride = fx.grid_dim.x * fx.Int32(K2_BLOCK)
        topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)
        c_topk = fx.Int32(topk)
        c_one = fx.Int32(1)
        c_oob = fx.Int32(0x7FFFFFFF)

        total = i32_tokens * c_topk

        _s = arith.index(0)
        _e = ArithValue(i32_niters).index_cast(T.index)
        _one = arith.index(1)
        for _i in range(_s, _e, _one):
            flat = gid + fx.Int32(_i) * stride
            valid = flat < total
            safe_flat = valid.select(flat, c_zero)
            token_id = safe_flat // c_topk
            topk_slot = safe_flat % c_topk
            eid = buffer_ops.buffer_load(topk_rsrc, safe_flat, vec_width=1, dtype=T.i32)
            # uint8 mesh: byte_offset = eid * mesh_stride + token_id
            byte_offset = eid * i32_mesh_stride + token_id
            val_i8 = arith.trunci(T.i8, topk_slot + c_one)
            # OOB offset for invalid threads (GPU silently drops OOB stores)
            safe_byte_off = valid.select(byte_offset, c_oob)
            buffer_ops.buffer_store(val_i8, ws_rsrc, safe_byte_off, offset_is_bytes=True)

    @flyc.jit
    def launch_p0(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_niters: fx.Int32,
        n_grid: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = p0_scatter_kernel(topk_ids, workspace, i32_tokens, i32_mesh_stride, i32_niters)
        launcher.launch(grid=(n_grid, 1, 1), block=(K2_BLOCK, 1, 1), stream=stream)

    # --- K3: P1 count kernel -------------------------------------------------
    # 256 threads (4 waves), vec_width=4: each thread loads 4 i32 words (16
    # mesh cells) per iteration.  4 waves provide 4x memory-level parallelism
    # vs the old 1-wave (64-thread) design, matching CK P1's block size.
    # Cross-warp reduction via LDS (4 partial sums, one per warp).
    K3_BLOCK = 256
    K3_NUM_WAVES = K3_BLOCK // WARP_SIZE
    K3_VEC_WIDTH = 4
    K3_WORDS_PER_ITER = K3_BLOCK * K3_VEC_WIDTH
    K3_WORDS_PER_ITER_LOG2 = (K3_WORDS_PER_ITER).bit_length() - 1

    k3_allocator = SmemAllocator(None, arch=arch)
    k3_reduce_offset = k3_allocator._align(k3_allocator.ptr, 16)
    k3_allocator.ptr = k3_reduce_offset + K3_NUM_WAVES * 4

    @flyc.kernel
    def p1_count_kernel(
        workspace: fx.Tensor,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
    ):
        eid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_ff = fx.Int32(0xFF)

        base_ptr = k3_allocator.get_base()
        reduce_mr = SmemPtr(base_ptr, k3_reduce_offset, T.i32, shape=(K3_NUM_WAVES,)).get()

        mesh_row_i32_base = (eid * i32_mesh_stride) >> fx.Int32(2)
        i32_words_per_row = i32_mesh_stride >> fx.Int32(2)
        n_iters = (i32_words_per_row + fx.Int32(K3_WORDS_PER_ITER - 1)) >> fx.Int32(K3_WORDS_PER_ITER_LOG2)

        for _i, state in range(arith.index(0), ArithValue(n_iters).index_cast(T.index),
                                arith.index(1), init=[c_zero]):
            cnt_so_far = state[0]

            word_base = fx.Int32(_i) * fx.Int32(K3_WORDS_PER_ITER) + tid * fx.Int32(K3_VEC_WIDTH)
            valid = word_base < i32_words_per_row
            safe_addr = mesh_row_i32_base + valid.select(word_base, c_zero)
            vec4 = buffer_ops.buffer_load(ws_rsrc, safe_addr, vec_width=4, dtype=T.i32)

            iter_cnt = c_zero
            for _wi in range_constexpr(K3_VEC_WIDTH):
                word = fx.Int32(fly_vector.extract(vec4, static_position=[_wi], dynamic_position=[]))
                word_valid = valid & ((word_base + fx.Int32(_wi)) < i32_words_per_row)
                b0 = word & c_ff
                b1 = (word >> fx.Int32(8)) & c_ff
                b2 = (word >> fx.Int32(16)) & c_ff
                b3 = (word >> fx.Int32(24)) & c_ff
                nz0 = word_valid.select((b0 != c_zero).select(c_one, c_zero), c_zero)
                nz1 = word_valid.select((b1 != c_zero).select(c_one, c_zero), c_zero)
                nz2 = word_valid.select((b2 != c_zero).select(c_one, c_zero), c_zero)
                nz3 = word_valid.select((b3 != c_zero).select(c_one, c_zero), c_zero)
                iter_cnt = iter_cnt + nz0 + nz1 + nz2 + nz3

            new_cnt = cnt_so_far + iter_cnt
            results = yield [new_cnt]
        cnt = results

        # Intra-warp reduce via shuffle_xor
        width_ws = fx.Int32(WARP_SIZE)
        for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
            off = fx.Int32(1 << sh)
            peer = cnt.shuffle_xor(off, width_ws)
            cnt = cnt + peer

        # Cross-warp reduce via LDS: lane 0 of each warp writes partial sum
        is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, c_zero)
        _if_l0 = scf.IfOp(is_lane0)
        with _if_then(_if_l0):
            wave_ix = ArithValue(wave).index_cast(T.index)
            _lds_store_raw(reduce_mr, cnt, wave_ix)
        gpu.barrier()

        # Thread 0 sums all warp partials and writes to HBM
        is_t0 = (tid == c_zero)
        total = c_zero
        for _w in range_constexpr(K3_NUM_WAVES):
            total = total + _lds_load_raw(reduce_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))

        cs_offset = i32_mesh_size + eid
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        safe_cs = is_t0.select(cs_offset, c_oob_idx)
        buffer_ops.buffer_store(total, ws_rsrc, safe_cs)

    @flyc.jit
    def launch_p1(
        workspace: fx.Tensor,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k3_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            k3_allocator.finalize()

        launcher = p1_count_kernel(workspace, i32_mesh_stride, i32_mesh_size)
        launcher.launch(grid=(E, 1, 1), block=(K3_BLOCK, 1, 1), stream=stream)

    # --- P0_v2: Fused clear+scatter+count kernel (for small T) ---------------
    # Replaces K1+K2+K3 with a single kernel launch.
    # Grid: E blocks (one per expert), Block: 512 threads (matching CK P0_v2).
    # Phase 1: clear this expert's mesh row
    # Phase 2: scan all T*topk assignments, filter by expert, byte stores
    # Phase 3: popcount + warp reduce + cross-wave LDS reduce -> expert_cumsum
    P0V2_BLOCK = 512
    P0V2_NUM_WAVES = P0V2_BLOCK // WARP_SIZE

    # Power-of-2 topk: use shift to avoid division
    _p0v2_topk_is_po2 = (topk & (topk - 1)) == 0 and topk > 0
    _p0v2_topk_log2 = topk.bit_length() - 1 if _p0v2_topk_is_po2 else 0

    # LDS for cross-wave reduction (same layout as K3)
    p0v2_allocator = SmemAllocator(None, arch=arch)
    p0v2_reduce_offset = p0v2_allocator._align(p0v2_allocator.ptr, 16)
    p0v2_allocator.ptr = p0v2_reduce_offset + P0V2_NUM_WAVES * 4

    @flyc.kernel(known_block_size=[P0V2_BLOCK, 1, 1])
    def p0v2_kernel(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
    ):
        eid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_ff = fx.Int32(0xFF)
        c_topk = fx.Int32(topk)
        c_block = fx.Int32(P0V2_BLOCK)

        base_ptr = p0v2_allocator.get_base()
        reduce_mr = SmemPtr(base_ptr, p0v2_reduce_offset, T.i32, shape=(P0V2_NUM_WAVES,)).get()

        # Precompute mesh row base (in i32 words) and words per row
        mesh_row_i32_base = (eid * i32_mesh_stride) >> fx.Int32(2)
        i32_words_per_row = i32_mesh_stride >> fx.Int32(2)

        # ---- Phase 1: Clear this expert's mesh row ----
        clear_niters = (i32_words_per_row + fx.Int32(P0V2_BLOCK - 1)) >> fx.Int32(9)  # ceil div by 512
        for _ci in range(arith.index(0), ArithValue(clear_niters).index_cast(T.index), arith.index(1)):
            word_idx = fx.Int32(_ci) * c_block + tid
            valid = word_idx < i32_words_per_row
            safe_idx = mesh_row_i32_base + valid.select(word_idx, c_zero)
            buffer_ops.buffer_store(c_zero, ws_rsrc, valid.select(safe_idx, fx.Int32(0x7FFFFFFF)))

        gpu.barrier()

        # ---- Phase 2: Scatter (scan all T*topk, filter by expert) ----
        # Each block scans ALL T*topk assignments, writes only matching expert's
        # tokens via byte store (same as K2). OOB index for non-matching threads.
        c_oob = fx.Int32(0x7FFFFFFF)
        total_assignments = i32_tokens * c_topk
        scatter_niters = (total_assignments + fx.Int32(P0V2_BLOCK - 1)) >> fx.Int32(9)
        for _si in range(arith.index(0), ArithValue(scatter_niters).index_cast(T.index), arith.index(1)):
            flat = fx.Int32(_si) * c_block + tid
            valid = flat < total_assignments
            safe_flat = valid.select(flat, c_zero)

            token_id = safe_flat >> fx.Int32(_p0v2_topk_log2) if _p0v2_topk_is_po2 else safe_flat // c_topk
            topk_slot = safe_flat & fx.Int32(topk - 1) if _p0v2_topk_is_po2 else safe_flat % c_topk

            expert_id = buffer_ops.buffer_load(topk_rsrc, safe_flat, vec_width=1, dtype=T.i32)

            is_mine = valid & (expert_id == eid)
            byte_offset = eid * i32_mesh_stride + token_id
            val_i8 = arith.trunci(T.i8, is_mine.select(topk_slot + c_one, c_zero))
            safe_byte_off = is_mine.select(byte_offset, c_zero)
            buffer_ops.buffer_store(val_i8, ws_rsrc, safe_byte_off, offset_is_bytes=True)

        gpu.barrier()

        # ---- Phase 3: Count non-zero bytes + warp/cross-wave reduce ----
        count_niters = (i32_words_per_row + fx.Int32(P0V2_BLOCK - 1)) >> fx.Int32(9)  # ceil div by 512
        for _ki, state in range(arith.index(0), ArithValue(count_niters).index_cast(T.index),
                                arith.index(1), init=[c_zero]):
            cnt_so_far = state[0]

            word_base = fx.Int32(_ki) * c_block + tid
            valid = word_base < i32_words_per_row
            safe_addr = mesh_row_i32_base + valid.select(word_base, c_zero)
            word = buffer_ops.buffer_load(ws_rsrc, safe_addr, vec_width=1, dtype=T.i32)

            b0 = word & c_ff
            b1 = (word >> fx.Int32(8)) & c_ff
            b2 = (word >> fx.Int32(16)) & c_ff
            b3 = (word >> fx.Int32(24)) & c_ff
            nz0 = valid.select((b0 != c_zero).select(c_one, c_zero), c_zero)
            nz1 = valid.select((b1 != c_zero).select(c_one, c_zero), c_zero)
            nz2 = valid.select((b2 != c_zero).select(c_one, c_zero), c_zero)
            nz3 = valid.select((b3 != c_zero).select(c_one, c_zero), c_zero)
            iter_cnt = nz0 + nz1 + nz2 + nz3

            new_cnt = cnt_so_far + iter_cnt
            results = yield [new_cnt]
        cnt = results

        # Intra-warp reduce via shuffle_xor
        width_ws = fx.Int32(WARP_SIZE)
        for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
            off = fx.Int32(1 << sh)
            peer = cnt.shuffle_xor(off, width_ws)
            cnt = cnt + peer

        # Cross-warp reduce via LDS: lane 0 of each warp writes partial sum
        is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, c_zero)
        _if_l0 = scf.IfOp(is_lane0)
        with _if_then(_if_l0):
            wave_ix = ArithValue(wave).index_cast(T.index)
            _lds_store_raw(reduce_mr, cnt, wave_ix)
        gpu.barrier()

        # Thread 0 sums all warp partials and writes to HBM
        is_t0 = (tid == c_zero)
        total = c_zero
        for _w in range_constexpr(P0V2_NUM_WAVES):
            total = total + _lds_load_raw(reduce_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))

        cs_offset = i32_mesh_size + eid
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        safe_cs = is_t0.select(cs_offset, c_oob_idx)
        buffer_ops.buffer_store(total, ws_rsrc, safe_cs)

    @flyc.jit
    def launch_p0v2(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        p0v2_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            p0v2_allocator.finalize()

        launcher = p0v2_kernel(topk_ids, workspace, i32_tokens, i32_mesh_stride, i32_mesh_size)
        launcher.launch(grid=(E, 1, 1), block=(P0V2_BLOCK, 1, 1), stream=stream)

    # --- K4: P23 prefix-sum + scatter + moe_buf zeroing ---------------------
    # Parallel design (matching CK P23): each block [0, E) independently
    # computes the SAME prefix sum, then scatters ONLY for expert blockIdx.x.
    # No inter-block barrier needed — redundant prefix sums are deterministic.
    K4_BLOCK = 256

    # LDS: cumsum[E+1] for prefix sums + cross-wave scratch for DPP scan
    K4_NUM_WAVES = K4_BLOCK // WARP_SIZE  # 4
    k4_allocator = SmemAllocator(None, arch=arch)
    k4_smem_cols = E + 1
    k4_cumsum_offset = k4_allocator._align(k4_allocator.ptr, 16)
    k4_allocator.ptr = k4_cumsum_offset + k4_smem_cols * 4
    k4_scatter_offset = k4_allocator._align(k4_allocator.ptr, 16)
    k4_allocator.ptr = k4_scatter_offset + K4_NUM_WAVES * 4

    # DPP constants (same as decode)
    DPP_ROW_SHR_1 = 0x111
    DPP_ROW_SHR_2 = 0x112
    DPP_ROW_SHR_4 = 0x114
    DPP_ROW_SHR_8 = 0x118
    DPP_ROW_MASK = 0xF
    DPP_BANK_MASK = 0xF

    @flyc.kernel
    def p23_kernel(
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c4 = fx.Int32(4)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_topk = fx.Int32(topk)
        c_sentinel = fx.Int32(topk << 24)
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        c_ff = fx.Int32(0xFF)

        # Buffer resources
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(topk_weights_tensor, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights_out, max_size=True)

        # LDS: cumsum[E+1] for prefix sums + cross-wave scratch
        base_ptr = k4_allocator.get_base()
        cumsum_mr = SmemPtr(base_ptr, k4_cumsum_offset, T.i32, shape=(k4_smem_cols,)).get()
        scatter_mr = SmemPtr(base_ptr, k4_scatter_offset, T.i32, shape=(K4_NUM_WAVES,)).get()

        is_sort_block = arith.cmpi(arith.CmpIPredicate.slt, bid, c_E)
        is_zero_block = arith.cmpi(arith.CmpIPredicate.sge, bid, c_E)

        # ================ MOE_BUF ZEROING (blocks >= E) ==================
        _if_zero = scf.IfOp(is_zero_block)
        with _if_then(_if_zero):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            zero_base_bid = bid - c_E
            zero_gid = zero_base_bid * fx.Int32(K4_BLOCK) + tid
            num_zero_blocks = fx.grid_dim.x - c_E
            zero_stride = num_zero_blocks * fx.Int32(K4_BLOCK)
            zero_niters = (i32_moe_buf_elems + zero_stride - c_one) // zero_stride
            _zs = arith.index(0)
            _ze = ArithValue(zero_niters).index_cast(T.index)
            _z1 = arith.index(1)
            for _z in range(_zs, _ze, _z1):
                z_idx = zero_gid + fx.Int32(_z) * zero_stride
                z_valid = z_idx < i32_moe_buf_elems
                buffer_ops.buffer_store(c_zero, moe_buf_rsrc, z_valid.select(z_idx, c_zero))

        # ================ PARALLEL PREFIX-SUM + MESH SCATTER (blocks 0..E-1) ==
        # Each block independently: prefix sum (redundant), scatter for its expert only.
        _if_sort = scf.IfOp(is_sort_block)
        with _if_then(_if_sort):
            my_expert = bid

            # Step 1: Load expert counts from workspace -> pad to unit_size -> LDS cumsum
            # K4_BLOCK == E == 256: thread tid loads expert tid's count.
            ws_cs_addr = i32_mesh_size + tid
            raw_cnt = buffer_ops.buffer_load(ws_rsrc, ws_cs_addr, vec_width=1, dtype=T.i32)
            blocks = (raw_cnt + c_unit - c_one) >> fx.Int32(5)  # // 32
            padded = (raw_cnt == c_zero).select(c_zero, blocks * c_unit)
            # Write padded count to cumsum[tid+1]; thread 0 also writes cumsum[0]=0
            _lds_store_raw(cumsum_mr, padded, ArithValue(tid + c_one).index_cast(T.index))
            is_t0_init = arith.cmpi(arith.CmpIPredicate.eq, tid, c_zero)
            _if_init_cs = scf.IfOp(is_t0_init)
            with _if_then(_if_init_cs):
                _lds_store_raw(cumsum_mr, c_zero, ArithValue(c_zero).index_cast(T.index))
            gpu.barrier()

            # Step 2: DPP inclusive prefix sum over cumsum LDS (all 256 threads, 4 waves)
            val = _lds_load_raw(cumsum_mr, ArithValue(tid + c_one).index_cast(T.index))
            val_raw = _unwrap_raw(val)
            zero_raw = _unwrap_raw(c_zero)

            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= c_one).select(val + fx.Int32(remote), val)

            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(2)).select(val + fx.Int32(remote), val)

            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(4)).select(val + fx.Int32(remote), val)

            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_8, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(8)).select(val + fx.Int32(remote), val)

            if WARP_SIZE > 16:
                src_lane_16 = (lane & fx.Int32(0x30)) - c_one
                src_addr_16 = src_lane_16 * c4
                remote16 = fly_rocdl.ds_bpermute(T.i32, src_addr_16, val)
                val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

            if WARP_SIZE > 32:
                src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
                src_addr_32 = src_lane_32 * c4
                remote32 = fly_rocdl.ds_bpermute(T.i32, src_addr_32, val)
                val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

            # val now holds intra-wave inclusive prefix sum.
            # Cross-wave accumulation via LDS scratch (ds_bpermute is wave-local).
            is_last_lane_ps = arith.cmpi(arith.CmpIPredicate.eq, lane, fx.Int32(WARP_SIZE - 1))
            _if_ll_ps = scf.IfOp(is_last_lane_ps)
            with _if_then(_if_ll_ps):
                _lds_store_raw(scatter_mr, val, ArithValue(wave).index_cast(T.index))
            gpu.barrier()
            cross_offset = c_zero
            for _w in range_constexpr(K4_NUM_WAVES - 1):
                w_total = _lds_load_raw(scatter_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))
                cross_offset = (wave > fx.Int32(_w)).select(cross_offset + w_total, cross_offset)
            total_padded = c_zero
            for _w in range_constexpr(K4_NUM_WAVES):
                total_padded = total_padded + _lds_load_raw(scatter_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))

            # Write inclusive prefix sum back to cumsum LDS
            inclusive_prefix = val + cross_offset
            _lds_store_raw(cumsum_mr, inclusive_prefix, ArithValue(tid + c_one).index_cast(T.index))
            gpu.barrier()

            # Read my_start and my_end from cumsum LDS
            my_start = _lds_load_raw(cumsum_mr, ArithValue(my_expert).index_cast(T.index))
            my_end = _lds_load_raw(cumsum_mr, ArithValue(my_expert + c_one).index_cast(T.index))

            # Block 0, thread 0 writes num_valid_ids
            is_b0 = arith.cmpi(arith.CmpIPredicate.eq, bid, c_zero)
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, c_zero)
            is_b0_t0 = is_b0 & is_t0
            _if_nv = scf.IfOp(is_b0_t0)
            with _if_then(_if_nv):
                nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
                buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, nvalid_rsrc, c_one)

            # Step 3: Write sorted_expert_ids for THIS expert (parallel across threads)
            sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
            blk_start = my_start >> fx.Int32(5)   # // 32 (unit_size)
            blk_end = my_end >> fx.Int32(5)       # // 32
            n_blks = blk_end - blk_start
            n_eid_iters = (n_blks + fx.Int32(K4_BLOCK) - c_one) >> fx.Int32(8)  # // 256
            for _eii in range(arith.index(0), ArithValue(n_eid_iters).index_cast(T.index), arith.index(1)):
                blk_idx = blk_start + fx.Int32(_eii) * fx.Int32(K4_BLOCK) + tid
                buffer_ops.buffer_store(my_expert, sorted_e_rsrc,
                                        (blk_idx < blk_end).select(blk_idx, c_oob_idx))

            # Step 4: Mesh-based scatter — read uint8 mesh from HBM, extract tokens,
            # DPP prefix sum over counts, cross-wave LDS reduction, scatter stores.
            i32_words_per_row = i32_mesh_stride >> fx.Int32(2)
            n_mesh_iters = (i32_words_per_row + fx.Int32(K4_BLOCK - 1)) >> fx.Int32(8)
            mesh_row_i32_base = (my_expert * i32_mesh_stride) >> fx.Int32(2)

            for _si, state in range(arith.index(0), ArithValue(n_mesh_iters).index_cast(T.index),
                                    arith.index(1), init=[my_start]):
                position = state[0]

                word_idx = fx.Int32(_si) * fx.Int32(K4_BLOCK) + tid
                col_valid = word_idx < i32_words_per_row
                safe_word_idx = col_valid.select(word_idx, c_zero)
                word = buffer_ops.buffer_load(ws_rsrc, mesh_row_i32_base + safe_word_idx,
                                              vec_width=1, dtype=T.i32)

                # Extract 4 bytes from the i32 word
                x0 = word & c_ff
                x1 = (word >> fx.Int32(8)) & c_ff
                x2 = (word >> fx.Int32(16)) & c_ff
                x3 = (word >> fx.Int32(24)) & c_ff
                base_col = word_idx * c4

                h0 = col_valid & (x0 != c_zero)
                h1 = col_valid & (x1 != c_zero)
                h2 = col_valid & (x2 != c_zero)
                h3 = col_valid & (x3 != c_zero)

                my_cnt = (h0.select(c_one, c_zero) + h1.select(c_one, c_zero)
                          + h2.select(c_one, c_zero) + h3.select(c_one, c_zero))

                # DPP inclusive prefix sum over my_cnt within each wave
                cnt_raw = _unwrap_raw(my_cnt)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                              DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True)
                my_cnt = (lane >= c_one).select(my_cnt + fx.Int32(remote), my_cnt)

                cnt_raw = _unwrap_raw(my_cnt)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                              DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True)
                my_cnt = (lane >= fx.Int32(2)).select(my_cnt + fx.Int32(remote), my_cnt)

                cnt_raw = _unwrap_raw(my_cnt)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                              DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True)
                my_cnt = (lane >= fx.Int32(4)).select(my_cnt + fx.Int32(remote), my_cnt)

                cnt_raw = _unwrap_raw(my_cnt)
                remote = fly_rocdl.update_dpp(T.i32, zero_raw, cnt_raw,
                                              DPP_ROW_SHR_8, DPP_ROW_MASK, DPP_BANK_MASK, True)
                my_cnt = (lane >= fx.Int32(8)).select(my_cnt + fx.Int32(remote), my_cnt)

                if WARP_SIZE > 16:
                    src_lane_16 = (lane & fx.Int32(0x30)) - c_one
                    src_addr_16 = src_lane_16 * c4
                    remote16 = fly_rocdl.ds_bpermute(T.i32, src_addr_16, my_cnt)
                    my_cnt = (lane >= fx.Int32(16)).select(my_cnt + fx.Int32(remote16), my_cnt)

                if WARP_SIZE > 32:
                    src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
                    src_addr_32 = src_lane_32 * c4
                    remote32 = fly_rocdl.ds_bpermute(T.i32, src_addr_32, my_cnt)
                    my_cnt = (lane >= fx.Int32(32)).select(my_cnt + fx.Int32(remote32), my_cnt)

                # my_cnt is now intra-wave inclusive prefix sum of per-thread token counts.
                # Cross-wave reduction via LDS scratch.
                is_last_lane_sc = arith.cmpi(arith.CmpIPredicate.eq, lane, fx.Int32(WARP_SIZE - 1))
                _if_ll_sc = scf.IfOp(is_last_lane_sc)
                with _if_then(_if_ll_sc):
                    _lds_store_raw(scatter_mr, my_cnt, ArithValue(wave).index_cast(T.index))
                gpu.barrier()

                wave_offset = c_zero
                for _w in range_constexpr(K4_NUM_WAVES - 1):
                    w_total = _lds_load_raw(scatter_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))
                    wave_offset = (wave > fx.Int32(_w)).select(wave_offset + w_total, wave_offset)
                batch_total = c_zero
                for _w in range_constexpr(K4_NUM_WAVES):
                    batch_total = batch_total + _lds_load_raw(scatter_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))
                gpu.barrier()

                # Convert to exclusive prefix: my_exclusive = my_cnt - my_thread_count
                my_thread_count = (h0.select(c_one, c_zero) + h1.select(c_one, c_zero)
                                   + h2.select(c_one, c_zero) + h3.select(c_one, c_zero))
                my_exclusive = my_cnt - my_thread_count + wave_offset

                # Scatter each valid token using predicated stores (OOB drops silently)
                scatter_base = position + my_exclusive

                # Token 0 (byte 0)
                token_id_0 = base_col
                topk_slot_0 = h0.select(x0 - c_one, c_zero)
                pid_0 = (topk_slot_0 << fx.Int32(24)) | token_id_0
                safe_slot_0 = h0.select(scatter_base, c_oob_idx)
                buffer_ops.buffer_store(pid_0, sorted_ids_rsrc, safe_slot_0)
                w_addr_0 = h0.select(token_id_0 * c_topk + topk_slot_0, c_zero)
                w_val_0 = buffer_ops.buffer_load(weights_rsrc, w_addr_0, vec_width=1, dtype=T.i32)
                buffer_ops.buffer_store(w_val_0, sorted_w_rsrc, safe_slot_0)

                # Token 1 (byte 1)
                off1 = scatter_base + h0.select(c_one, c_zero)
                token_id_1 = base_col + c_one
                topk_slot_1 = h1.select(x1 - c_one, c_zero)
                pid_1 = (topk_slot_1 << fx.Int32(24)) | token_id_1
                safe_slot_1 = h1.select(off1, c_oob_idx)
                buffer_ops.buffer_store(pid_1, sorted_ids_rsrc, safe_slot_1)
                w_addr_1 = h1.select(token_id_1 * c_topk + topk_slot_1, c_zero)
                w_val_1 = buffer_ops.buffer_load(weights_rsrc, w_addr_1, vec_width=1, dtype=T.i32)
                buffer_ops.buffer_store(w_val_1, sorted_w_rsrc, safe_slot_1)

                # Token 2 (byte 2)
                off2 = off1 + h1.select(c_one, c_zero)
                token_id_2 = base_col + fx.Int32(2)
                topk_slot_2 = h2.select(x2 - c_one, c_zero)
                pid_2 = (topk_slot_2 << fx.Int32(24)) | token_id_2
                safe_slot_2 = h2.select(off2, c_oob_idx)
                buffer_ops.buffer_store(pid_2, sorted_ids_rsrc, safe_slot_2)
                w_addr_2 = h2.select(token_id_2 * c_topk + topk_slot_2, c_zero)
                w_val_2 = buffer_ops.buffer_load(weights_rsrc, w_addr_2, vec_width=1, dtype=T.i32)
                buffer_ops.buffer_store(w_val_2, sorted_w_rsrc, safe_slot_2)

                # Token 3 (byte 3)
                off3 = off2 + h2.select(c_one, c_zero)
                token_id_3 = base_col + fx.Int32(3)
                topk_slot_3 = h3.select(x3 - c_one, c_zero)
                pid_3 = (topk_slot_3 << fx.Int32(24)) | token_id_3
                safe_slot_3 = h3.select(off3, c_oob_idx)
                buffer_ops.buffer_store(pid_3, sorted_ids_rsrc, safe_slot_3)
                w_addr_3 = h3.select(token_id_3 * c_topk + topk_slot_3, c_zero)
                w_val_3 = buffer_ops.buffer_load(weights_rsrc, w_addr_3, vec_width=1, dtype=T.i32)
                buffer_ops.buffer_store(w_val_3, sorted_w_rsrc, safe_slot_3)

                pos_next = position + batch_total
                results = yield [pos_next]
            scatter_end_pos_t0 = results

            # Step 5: Fill padding with sentinel for THIS expert (parallel)
            sentinel_val = c_sentinel | i32_tokens
            pad_count = my_end - scatter_end_pos_t0
            pad_niters = (pad_count + fx.Int32(K4_BLOCK) - c_one) >> fx.Int32(8)  # // 256
            for _pi in range(arith.index(0), ArithValue(pad_niters).index_cast(T.index), arith.index(1)):
                pad_slot = scatter_end_pos_t0 + fx.Int32(_pi) * fx.Int32(K4_BLOCK) + tid
                pad_valid = pad_slot < my_end
                buffer_ops.buffer_store(sentinel_val, sorted_ids_rsrc,
                                        pad_valid.select(pad_slot, c_oob_idx))
                buffer_ops.buffer_store(c_zero, sorted_w_rsrc,
                                        pad_valid.select(pad_slot, c_oob_idx))

    @flyc.jit
    def launch_p23(
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        k4_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            k4_allocator.finalize()

        launcher = p23_kernel(
            workspace, topk_weights_tensor,
            sorted_token_ids, sorted_weights_out, sorted_expert_ids,
            num_valid_ids_out, moe_buf,
            i32_tokens, i32_mesh_stride, i32_mesh_size, i32_moe_buf_elems,
        )
        launcher.launch(grid=(n_grid, 1, 1), block=(K4_BLOCK, 1, 1), stream=stream)

    return launch_clear_ws, launch_p0, launch_p1, launch_p23, launch_p0v2


# ---------------------------------------------------------------------------
# Meshless sort: single kernel, no HBM mesh, for small T (T<=512)
# Counts in registers, prefix sum in LDS (reusing K4's proven pattern),
# scatter via LDS atomicAdd. 1 launch instead of P0v2+K4's 2.
# ---------------------------------------------------------------------------
_meshless_cf_cache = {}  # (num_experts, topk, unit_size, n_grid) -> CompiledFunction


@functools.lru_cache(maxsize=256)
def compile_meshless_sort(*, num_experts, topk, unit_size=UNIT_SIZE):
    """Compile the single-kernel meshless MoE sorting kernel.

    Replaces P0v2 + K4 for small T (<=512) with a single kernel launch.
    No HBM workspace needed — all work happens in registers + LDS.

    Algorithm:
      Phase 1: Each thread (=expert tid) counts its tokens by scanning topk_ids
      Phase 2: LDS cumsum[E+1] + DPP prefix sum (same proven pattern as K4)
      Phase 3: Expert IDs + num_valid_ids
      Phase 4: Atomic scatter — re-scan topk_ids, LDS atomicAdd for position
      Phase 5: Padding + moe_buf zeroing

    Grid: E + n_zero_blocks
    Block: 256 threads (1 thread per expert for E=256)
    LDS: cumsum[E+1] i32 + 1 i32 atomic counter
    """
    arch = get_hip_arch()
    WARP_SIZE = get_warp_size(arch)
    E = num_experts
    ML_BLOCK = 256  # Must equal E for 1:1 thread-expert mapping
    ML_NUM_WAVES = ML_BLOCK // WARP_SIZE

    _ml_topk_is_po2 = (topk & (topk - 1)) == 0 and topk > 0
    _ml_topk_log2 = topk.bit_length() - 1 if _ml_topk_is_po2 else 0

    # DPP constants (same as K4)
    DPP_ROW_SHR_1 = 0x111
    DPP_ROW_SHR_2 = 0x112
    DPP_ROW_SHR_4 = 0x114
    DPP_ROW_SHR_8 = 0x118
    DPP_ROW_MASK = 0xF
    DPP_BANK_MASK = 0xF

    # LDS: cumsum[E+1] + scratch[ML_NUM_WAVES] (cross-wave) + atomic counter (1 i32)
    ml_allocator = SmemAllocator(None, arch=arch)
    ml_cumsum_offset = ml_allocator._align(ml_allocator.ptr, 16)
    ml_allocator.ptr = ml_cumsum_offset + (E + 1) * 4
    ml_scratch_offset = ml_allocator._align(ml_allocator.ptr, 16)
    ml_allocator.ptr = ml_scratch_offset + ML_NUM_WAVES * 4
    ml_atomic_offset = ml_allocator._align(ml_allocator.ptr, 4)
    ml_allocator.ptr = ml_atomic_offset + 4

    @flyc.kernel
    def meshless_sort_kernel(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c4 = fx.Int32(4)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_topk = fx.Int32(topk)
        c_sentinel = fx.Int32(topk << 24)
        c_oob_idx = fx.Int32(0x7FFFFFFF)

        topk_rsrc = buffer_ops.create_buffer_resource(topk_ids_tensor, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(topk_weights_tensor, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights_out, max_size=True)

        base_ptr = ml_allocator.get_base()
        cumsum_mr = SmemPtr(base_ptr, ml_cumsum_offset, T.i32, shape=(E + 1,)).get()
        scratch_mr = SmemPtr(base_ptr, ml_scratch_offset, T.i32, shape=(ML_NUM_WAVES,)).get()
        atomic_mr = SmemPtr(base_ptr, ml_atomic_offset, T.i32, shape=(1,)).get()

        is_sort_block = arith.cmpi(arith.CmpIPredicate.slt, bid, c_E)
        is_zero_block = arith.cmpi(arith.CmpIPredicate.sge, bid, c_E)

        # ================ MOE_BUF ZEROING (blocks >= E) ==================
        _if_zero = scf.IfOp(is_zero_block)
        with _if_then(_if_zero):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            zero_base_bid = bid - c_E
            zero_gid = zero_base_bid * fx.Int32(ML_BLOCK) + tid
            num_zero_blocks = fx.grid_dim.x - c_E
            zero_stride = num_zero_blocks * fx.Int32(ML_BLOCK)
            zero_niters = (i32_moe_buf_elems + zero_stride - c_one) // zero_stride
            _zs = arith.index(0)
            _ze = ArithValue(zero_niters).index_cast(T.index)
            _z1 = arith.index(1)
            for _z in range(_zs, _ze, _z1):
                z_idx = zero_gid + fx.Int32(_z) * zero_stride
                z_valid = z_idx < i32_moe_buf_elems
                buffer_ops.buffer_store(c_zero, moe_buf_rsrc, z_valid.select(z_idx, c_zero))

        # ================ SORTING (blocks 0..E-1) ========================
        _if_sort = scf.IfOp(is_sort_block)
        with _if_then(_if_sort):
            my_expert = bid

            # ---- Phase 1: Cooperative counting via LDS histogram ----
            # All threads cooperatively scan T*topk assignments.
            # Each thread atomicAdds to LDS histogram[expert_id].
            # O(T*topk/ML_BLOCK) per thread instead of O(T*topk) serial.
            total_assignments = i32_tokens * c_topk

            # Zero the histogram (cumsum_mr[0..E-1] as histogram)
            tid_valid_for_hist = tid < c_E
            _lds_store_raw(cumsum_mr, c_zero,
                           ArithValue(tid_valid_for_hist.select(tid, c_zero)).index_cast(T.index))
            gpu.barrier()

            # Cooperative scan: each thread processes its stride of assignments
            one_raw = _unwrap_raw(c_one)
            n_count_iters = (total_assignments + fx.Int32(ML_BLOCK - 1)) >> fx.Int32(8)
            for _ci in range(arith.index(0), ArithValue(n_count_iters).index_cast(T.index),
                             arith.index(1)):
                flat = fx.Int32(_ci) * fx.Int32(ML_BLOCK) + tid
                valid = flat < total_assignments
                safe_flat = valid.select(flat, c_zero)
                eid = buffer_ops.buffer_load(topk_rsrc, safe_flat, vec_width=1, dtype=T.i32)
                # AtomicAdd to histogram[eid] for valid assignments
                safe_eid = valid.select(eid, c_zero)
                eid_ix = ArithValue(safe_eid).index_cast(T.index)
                valid_i1 = arith.cmpi(arith.CmpIPredicate.ne, valid.select(c_one, c_zero), c_zero)
                _if_valid = scf.IfOp(valid_i1)
                with _if_then(_if_valid):
                    memref_ops.atomic_rmw(arith.AtomicRMWKind.addi, one_raw,
                                          cumsum_mr, [eid_ix])
            gpu.barrier()

            # Read this thread's expert count from histogram
            my_count = tid_valid_for_hist.select(
                _lds_load_raw(cumsum_mr, ArithValue(tid).index_cast(T.index)),
                c_zero)

            # ---- Phase 2: Prefix sum (P0v2 pattern — all threads, cross-wave via scratch) ----
            # Write padded count to LDS cumsum[tid+1]; thread 0 writes cumsum[0]=0
            # Guard for E < ML_BLOCK: only threads tid < E write to cumsum
            blocks_for = (my_count + c_unit - c_one) >> fx.Int32(5)
            padded = (my_count == c_zero).select(c_zero, blocks_for * c_unit)
            tid_valid_for_ps = tid < c_E
            safe_cs_wr = tid_valid_for_ps.select(tid + c_one, c_zero)
            _lds_store_raw(cumsum_mr, tid_valid_for_ps.select(padded, c_zero),
                           ArithValue(safe_cs_wr).index_cast(T.index))
            is_t0_cs = arith.cmpi(arith.CmpIPredicate.eq, tid, c_zero)
            _if_init_cs = scf.IfOp(is_t0_cs)
            with _if_then(_if_init_cs):
                _lds_store_raw(cumsum_mr, c_zero, ArithValue(c_zero).index_cast(T.index))
            gpu.barrier()

            # DPP inclusive prefix sum (all threads, same as P0v2)
            # Threads with tid >= E load 0 (from cumsum[0])
            safe_cs_rd = tid_valid_for_ps.select(tid + c_one, c_zero)
            val = _lds_load_raw(cumsum_mr, ArithValue(safe_cs_rd).index_cast(T.index))
            val = tid_valid_for_ps.select(val, c_zero)
            val_raw = _unwrap_raw(val)
            zero_raw = _unwrap_raw(c_zero)

            # DPP row_shr 1
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= c_one).select(val + fx.Int32(remote), val)

            # DPP row_shr 2
            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(2)).select(val + fx.Int32(remote), val)

            # DPP row_shr 4
            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(4)).select(val + fx.Int32(remote), val)

            # DPP row_shr 8
            val_raw = _unwrap_raw(val)
            remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw,
                                          DPP_ROW_SHR_8, DPP_ROW_MASK, DPP_BANK_MASK, True)
            val = (lane >= fx.Int32(8)).select(val + fx.Int32(remote), val)

            if WARP_SIZE > 16:
                src_lane_16 = (lane & fx.Int32(0x30)) - c_one
                src_addr_16 = src_lane_16 * c4
                remote16 = fly_rocdl.ds_bpermute(T.i32, src_addr_16, val)
                val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

            if WARP_SIZE > 32:
                src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
                src_addr_32 = src_lane_32 * c4
                remote32 = fly_rocdl.ds_bpermute(T.i32, src_addr_32, val)
                val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

            # Cross-wave accumulation via LDS scratch
            is_last_lane_ps = arith.cmpi(arith.CmpIPredicate.eq, lane, fx.Int32(WARP_SIZE - 1))
            _if_ll_ps = scf.IfOp(is_last_lane_ps)
            with _if_then(_if_ll_ps):
                _lds_store_raw(scratch_mr, val, ArithValue(wave).index_cast(T.index))
            gpu.barrier()
            cross_offset = c_zero
            for _w in range_constexpr(ML_NUM_WAVES - 1):
                w_total = _lds_load_raw(scratch_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))
                cross_offset = (wave > fx.Int32(_w)).select(cross_offset + w_total, cross_offset)
            total_padded = c_zero
            for _w in range_constexpr(ML_NUM_WAVES):
                total_padded = total_padded + _lds_load_raw(scratch_mr, ArithValue(fx.Int32(_w)).index_cast(T.index))

            # Write inclusive prefix sum back to cumsum LDS (guarded for tid < E)
            inclusive_prefix = val + cross_offset
            _lds_store_raw(cumsum_mr, tid_valid_for_ps.select(inclusive_prefix, c_zero),
                           ArithValue(safe_cs_wr).index_cast(T.index))
            gpu.barrier()

            # Read expert range from cumsum
            my_start = _lds_load_raw(cumsum_mr, ArithValue(my_expert).index_cast(T.index))
            my_end = _lds_load_raw(cumsum_mr, ArithValue(my_expert + c_one).index_cast(T.index))

            # num_valid_ids (block 0 thread 0)
            is_b0 = arith.cmpi(arith.CmpIPredicate.eq, bid, c_zero)
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, c_zero)
            is_b0_t0 = is_b0 & is_t0
            _if_nv = scf.IfOp(is_b0_t0)
            with _if_then(_if_nv):
                nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
                buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, nvalid_rsrc, c_one)

            # ---- Phase 3: Write sorted_expert_ids ----
            sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
            blk_start = my_start >> fx.Int32(5)
            blk_end = my_end >> fx.Int32(5)
            n_blks = blk_end - blk_start
            n_eid_iters = (n_blks + fx.Int32(ML_BLOCK) - c_one) >> fx.Int32(8)
            for _eii in range(arith.index(0), ArithValue(n_eid_iters).index_cast(T.index), arith.index(1)):
                blk_idx = blk_start + fx.Int32(_eii) * fx.Int32(ML_BLOCK) + tid
                buffer_ops.buffer_store(my_expert, sorted_e_rsrc,
                                        (blk_idx < blk_end).select(blk_idx, c_oob_idx))

            # ---- Phase 4: Atomic scatter (no mesh — re-read topk_ids) ----
            # Initialize LDS counter to 0
            _if_t0 = scf.IfOp(is_t0)
            with _if_then(_if_t0):
                _lds_store_raw(atomic_mr, c_zero, ArithValue(c_zero).index_cast(T.index))
            gpu.barrier()

            zero_ix = ArithValue(c_zero).index_cast(T.index)
            one_raw = _unwrap_raw(c_one)

            n_scatter_iters = (total_assignments + fx.Int32(ML_BLOCK - 1)) >> fx.Int32(8)
            for _si in range(arith.index(0), ArithValue(n_scatter_iters).index_cast(T.index),
                             arith.index(1)):
                flat = fx.Int32(_si) * fx.Int32(ML_BLOCK) + tid
                valid_s = flat < total_assignments
                safe_flat_s = valid_s.select(flat, c_zero)
                # Decode token_id and topk_slot
                if const_expr(_ml_topk_is_po2):
                    token_id = safe_flat_s >> fx.Int32(_ml_topk_log2)
                    topk_slot = safe_flat_s & fx.Int32(topk - 1)
                    w_flat = (token_id << fx.Int32(_ml_topk_log2)) + topk_slot
                else:
                    token_id = safe_flat_s // c_topk
                    topk_slot = safe_flat_s - token_id * c_topk
                    w_flat = token_id * c_topk + topk_slot
                expert_id = buffer_ops.buffer_load(topk_rsrc, safe_flat_s, vec_width=1, dtype=T.i32)
                is_mine = valid_s & (expert_id == my_expert)
                # Pre-compute values outside IfOp
                pid = (topk_slot << fx.Int32(24)) | token_id
                w_val = buffer_ops.buffer_load(
                    weights_rsrc, valid_s.select(w_flat, c_zero), vec_width=1, dtype=T.i32)
                # Conditional scatter via scf.IfOp
                is_mine_i1 = arith.cmpi(arith.CmpIPredicate.ne, is_mine.select(c_one, c_zero), c_zero)
                _if_mine = scf.IfOp(is_mine_i1)
                with _if_then(_if_mine):
                    pos = fx.Int32(memref_ops.atomic_rmw(
                        arith.AtomicRMWKind.addi, one_raw, atomic_mr, [zero_ix]))
                    slot = my_start + pos
                    buffer_ops.buffer_store(pid, sorted_ids_rsrc, slot)
                    buffer_ops.buffer_store(w_val, sorted_w_rsrc, slot)
            gpu.barrier()

            # Read final scatter position from counter
            scatter_end_pos = my_start + _lds_load_raw(atomic_mr, zero_ix)

            # ---- Phase 5: Fill padding with sentinel ----
            sentinel_val = c_sentinel | i32_tokens
            pad_count = my_end - scatter_end_pos
            pad_niters = (pad_count + fx.Int32(ML_BLOCK) - c_one) >> fx.Int32(8)
            for _pi in range(arith.index(0), ArithValue(pad_niters).index_cast(T.index), arith.index(1)):
                pad_slot = scatter_end_pos + fx.Int32(_pi) * fx.Int32(ML_BLOCK) + tid
                pad_valid = pad_slot < my_end
                buffer_ops.buffer_store(sentinel_val, sorted_ids_rsrc,
                                        pad_valid.select(pad_slot, c_oob_idx))
                buffer_ops.buffer_store(c_zero, sorted_w_rsrc,
                                        pad_valid.select(pad_slot, c_oob_idx))

    @flyc.jit
    def launch_meshless_sort(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        ml_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            ml_allocator.finalize()

        launcher = meshless_sort_kernel(
            topk_ids_tensor, topk_weights_tensor,
            sorted_token_ids, sorted_weights_out, sorted_expert_ids,
            num_valid_ids_out, moe_buf,
            i32_tokens, i32_moe_buf_elems,
        )
        launcher.launch(grid=(n_grid, 1, 1), block=(ML_BLOCK, 1, 1), stream=stream)

    return launch_meshless_sort


# ---------------------------------------------------------------------------
# Host-side entry point
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _compute_sub_tokens(num_experts, arch=None):
    """Compute the LDS-capacity threshold (sub_tokens) for decode vs prefill decision.

    Returns the max T that fits in LDS for the decode (single-kernel) path.
    Same formula as compile_moe_sorting_decode.
    """
    if arch is None:
        arch = get_hip_arch()
    E = num_experts
    smem_cols = E + 1
    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536
    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        return 0  # LDS too small — always use prefill
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    # Cap at 24: beyond this, the prefill path (256 parallel blocks) outperforms
    # the decode path (1 block) even on GPUs with larger LDS (gfx950: 160KB).
    return min(r_for_sub, 24)


def moe_sorting_flydsl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim=4096,
    moebuf_dtype=None,
    topk=None,
    unit_size=UNIT_SIZE,
    max_tokens=None,
    sorted_ids=None,
    sorted_weights=None,
    sorted_expert_ids=None,
    num_valid_ids=None,
    moe_buf=None,
):
    """MoE sorting using FlyDSL kernel (decode + prefill paths).

    API matches aiter.fused_moe.moe_sorting for drop-in replacement.
    Pre-allocated output tensors can be passed to avoid per-call allocation
    overhead (~2 us savings in graph mode).

    Returns
    -------
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
    """
    import torch

    if moebuf_dtype is None:
        moebuf_dtype = torch.bfloat16
    if topk is None:
        topk = topk_ids.shape[1]
    M = topk_ids.shape[0]

    sub_tokens = _compute_sub_tokens(num_experts)

    max_num_tokens_padded = M * topk + num_experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size

    device = topk_ids.device
    if sorted_ids is None:
        sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    if sorted_weights is None:
        sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    if sorted_expert_ids is None:
        sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    if num_valid_ids is None:
        num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty(M * model_dim * 2 // 4, dtype=torch.int32, device=device)

    moe_buf_elems = moe_buf.shape[0]

    # Meshless threshold: single kernel for T <= 512 (when T > sub_tokens).
    # Avoids HBM mesh and saves one kernel launch vs P0v2+K4.
    MESHLESS_MAX_T = 512

    if M <= sub_tokens:
        # Decode path: single kernel
        if max_tokens is None:
            max_tokens = max(M, 8)
            max_tokens = ((max_tokens + 7) // 8) * 8
        assert M <= max_tokens, f"T={M} exceeds max_tokens={max_tokens}"

        n_zero_blocks = (moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_grid_blocks = 1 + n_zero_blocks

        launch_moe_sorting_decode_path(
            topk_ids, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_buf, M, moe_buf_elems, n_grid_blocks,
            num_experts=num_experts, topk=topk,
            max_tokens=max_tokens, unit_size=unit_size,
        )
    elif M <= MESHLESS_MAX_T and num_experts <= 256:
        # Meshless path: single kernel, no HBM workspace
        launch_meshless_sort_path(
            topk_ids, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_buf, M, moe_buf_elems,
            num_experts=num_experts, topk=topk, unit_size=unit_size,
        )
    else:
        # Prefill path: multiple kernels via HBM workspace
        # uint8 mesh: 1 byte per cell (topk_slot+1, max 9 for topk=8)
        mesh_stride = ((M + 31) // 32) * 32  # padded to 32 for byte alignment
        ws_mesh_bytes = num_experts * mesh_stride  # 1 byte per cell
        ws_mesh_i32 = (ws_mesh_bytes + 3) // 4  # i32 elements for clearing
        ws_total = ws_mesh_i32 + (num_experts + 1)  # mesh + expert_cumsum
        workspace = torch.empty(ws_total, dtype=torch.int32, device=device)

        launch_moe_sorting_prefill_path(
            topk_ids, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_buf, workspace,
            M, moe_buf_elems, mesh_stride, ws_mesh_i32, ws_total,
            num_experts=num_experts, topk=topk, unit_size=unit_size,
        )

    moe_buf_out = moe_buf.view(moebuf_dtype).reshape(M, model_dim)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf_out


def launch_moe_sorting_decode_path(
    topk_ids, topk_weights,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    moe_buf_i32, i32_tokens, i32_moe_buf_elems, n_grid_blocks,
    *,
    num_experts, topk, max_tokens=128, unit_size=UNIT_SIZE,
):
    """Low-level launcher for decode path: single kernel.

    This is the hot-path entry point — no torch ops, just JIT dispatch.
    Uses AOT-compiled dispatch after the first call to bypass the ~70 us
    JIT overhead (inspect.Signature.bind + cache key + dict lookup).
    """
    import torch

    cache_key = (num_experts, topk, max_tokens, unit_size, n_grid_blocks)
    cf = _decode_cf_cache.get(cache_key)
    if cf is not None:
        # Fast path: ~5 us (update ctypes slots + invoke C function pointer)
        stream = torch.cuda.current_stream()
        cf(topk_ids, topk_weights,
           sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
           moe_buf_i32,
           i32_tokens, i32_moe_buf_elems,
           n_grid_blocks,
           fx.Stream(stream))
        return

    # Cold path: first call triggers JIT compilation
    launch_fn = compile_moe_sorting_decode(
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        unit_size=unit_size,
    )
    stream = torch.cuda.current_stream()
    launch_fn(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf_i32,
        i32_tokens, i32_moe_buf_elems,
        n_grid_blocks,
        stream=stream,
    )

    # Build and cache the CompiledFunction for subsequent fast dispatch.
    # flyc.compile() re-invokes launch_fn (hits the CallState fast path)
    # and returns a CompiledFunction wrapping the pre-built CallState.
    cf = flyc.compile(
        launch_fn,
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf_i32,
        i32_tokens, i32_moe_buf_elems,
        n_grid_blocks,
        fx.Stream(stream),
    )
    _decode_cf_cache[cache_key] = cf


def launch_meshless_sort_path(
    topk_ids, topk_weights,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    moe_buf_i32, i32_tokens, i32_moe_buf_elems,
    *,
    num_experts, topk, unit_size=UNIT_SIZE,
):
    """Low-level launcher for meshless sort path: single kernel, no HBM workspace.

    For T <= 512: replaces P0v2+K4 with a single kernel.
    Uses AOT-compiled dispatch after the first call.
    """
    import torch

    try:
        from aiter.jit.utils.chip_info import get_cu_num
        NUM_CU = get_cu_num()
    except ImportError:
        NUM_CU = 304  # MI300X fallback

    ML_OCCUPANCY = 2
    n_zero_blocks = min((i32_moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, NUM_CU * ML_OCCUPANCY)
    n_grid = num_experts + n_zero_blocks

    cache_key = (num_experts, topk, unit_size, n_grid)
    cf = _meshless_cf_cache.get(cache_key)
    if cf is not None:
        stream = torch.cuda.current_stream()
        cf(topk_ids, topk_weights,
           sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
           moe_buf_i32,
           i32_tokens, i32_moe_buf_elems,
           n_grid,
           fx.Stream(stream))
        return

    launch_fn = compile_meshless_sort(
        num_experts=num_experts,
        topk=topk,
        unit_size=unit_size,
    )
    stream = torch.cuda.current_stream()
    launch_fn(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf_i32,
        i32_tokens, i32_moe_buf_elems,
        n_grid,
        stream=stream,
    )

    cf = flyc.compile(
        launch_fn,
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf_i32,
        i32_tokens, i32_moe_buf_elems,
        n_grid,
        fx.Stream(stream),
    )
    _meshless_cf_cache[cache_key] = cf


def launch_moe_sorting_prefill_path(
    topk_ids, topk_weights,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    moe_buf_i32, workspace,
    i32_tokens, i32_moe_buf_elems, mesh_stride, mesh_size, ws_total,
    *,
    num_experts, topk, unit_size=UNIT_SIZE,
):
    """Low-level launcher for prefill path via HBM workspace.

    For small T (<=512): fused P0_v2 (clear+scatter+count) + K4.
    For large T: 4 separate kernels K1+K2+K3+K4.

    Uses AOT-compiled dispatch after the first call for each sub-kernel
    to bypass JIT overhead.
    """
    import torch

    launch_clear_ws, launch_p0, launch_p1, launch_p23, launch_p0v2 = compile_moe_sorting_prefill(
        num_experts=num_experts,
        topk=topk,
        unit_size=unit_size,
    )

    stream = torch.cuda.current_stream()
    stream_arg = fx.Stream(stream)

    try:
        from aiter.jit.utils.chip_info import get_cu_num
        NUM_CU = get_cu_num()
    except ImportError:
        NUM_CU = 304  # MI300X fallback

    base_key = (num_experts, topk, unit_size)
    use_p0v2 = i32_tokens <= 512

    if use_p0v2:
        # P0_v2: no constexpr args
        ck = base_key + ("p0v2",)
        cf = _prefill_cf_cache.get(ck)
        if cf is not None:
            cf(topk_ids, workspace, i32_tokens, mesh_stride, mesh_size, stream_arg)
        else:
            launch_p0v2(topk_ids, workspace, i32_tokens, mesh_stride, mesh_size, stream=stream)
            cf = flyc.compile(launch_p0v2, topk_ids, workspace, i32_tokens, mesh_stride, mesh_size, stream_arg)
            _prefill_cf_cache[ck] = cf
    else:
        # K1: ClearWorkspace — constexpr n_grid
        k1_grid = (ws_total + 1023) // 1024
        ck = base_key + ("clear_ws", k1_grid)
        cf = _prefill_cf_cache.get(ck)
        if cf is not None:
            cf(workspace, ws_total, k1_grid, stream_arg)
        else:
            launch_clear_ws(workspace, ws_total, k1_grid, stream=stream)
            cf = flyc.compile(launch_clear_ws, workspace, ws_total, k1_grid, stream_arg)
            _prefill_cf_cache[ck] = cf

        # K2: P0 scatter — constexpr n_grid
        k2_grid = min(NUM_CU * 2, (i32_tokens * topk + 255) // 256)
        k2_total = i32_tokens * topk
        k2_stride = k2_grid * 256
        k2_niters = (k2_total + k2_stride - 1) // k2_stride
        ck = base_key + ("p0", k2_grid)
        cf = _prefill_cf_cache.get(ck)
        if cf is not None:
            cf(topk_ids, workspace, i32_tokens, mesh_stride, k2_niters, k2_grid, stream_arg)
        else:
            launch_p0(topk_ids, workspace, i32_tokens, mesh_stride, k2_niters, k2_grid, stream=stream)
            cf = flyc.compile(launch_p0, topk_ids, workspace, i32_tokens, mesh_stride, k2_niters, k2_grid, stream_arg)
            _prefill_cf_cache[ck] = cf

        # K3: P1 count — no constexpr args
        ck = base_key + ("p1",)
        cf = _prefill_cf_cache.get(ck)
        if cf is not None:
            cf(workspace, mesh_stride, mesh_size, stream_arg)
        else:
            launch_p1(workspace, mesh_stride, mesh_size, stream=stream)
            cf = flyc.compile(launch_p1, workspace, mesh_stride, mesh_size, stream_arg)
            _prefill_cf_cache[ck] = cf

    # K4: P23 prefix-sum + mesh scatter — constexpr n_grid
    K4_OCCUPANCY = 2
    n_zero_blocks = min((i32_moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, NUM_CU * K4_OCCUPANCY)
    k4_grid = num_experts + n_zero_blocks
    ck = base_key + ("p23", k4_grid)
    cf = _prefill_cf_cache.get(ck)
    if cf is not None:
        cf(workspace, topk_weights,
           sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
           moe_buf_i32,
           i32_tokens, mesh_stride, mesh_size, i32_moe_buf_elems,
           k4_grid,
           stream_arg)
    else:
        launch_p23(
            workspace, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_buf_i32,
            i32_tokens, mesh_stride, mesh_size, i32_moe_buf_elems,
            k4_grid,
            stream=stream,
        )
        cf = flyc.compile(
            launch_p23,
            workspace, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_buf_i32,
            i32_tokens, mesh_stride, mesh_size, i32_moe_buf_elems,
            k4_grid,
            stream_arg,
        )
        _prefill_cf_cache[ck] = cf


# Keep backward-compatible launch_moe_sorting for any external callers
def launch_moe_sorting(
    topk_ids, topk_weights,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    moe_buf_i32, i32_tokens, i32_moe_buf_elems, n_grid_blocks,
    *,
    num_experts, topk, max_tokens=128, unit_size=UNIT_SIZE,
):
    """Low-level launcher (backward-compatible): dispatches decode path.

    For prefill, use launch_moe_sorting_prefill_path() directly or
    let moe_sorting_flydsl() auto-dispatch.
    """
    launch_moe_sorting_decode_path(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf_i32, i32_tokens, i32_moe_buf_elems, n_grid_blocks,
        num_experts=num_experts, topk=topk,
        max_tokens=max_tokens, unit_size=unit_size,
    )
