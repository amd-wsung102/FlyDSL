# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TopK Gating Softmax kernel builder using the @flyc.kernel API.

Fuses softmax + top-K selection + optional renormalization for MoE gating:

  1. softmax(logits)  = exp(x - max(x)) / sum(exp(x - max(x)))
  2. top-K selection   = K iterations of argmax-then-mask
  3. renormalize       = rescale K selected weights to sum to 1.0

Uses exp2(x * log2e) for fast exponentiation.
Register-buffers the entire expert-logit row across passes.
Outputs: topk_weights (f32), topk_indices (i32), token_expert_indices (i32).

Generic scalar path for arbitrary num_experts.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size


KERNEL_NAME = "topk_gating_softmax_kernel"

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()


def build_topk_gating_softmax_module(
    num_experts: int,
    topk: int,
    dtype_str: str = "bf16",
    renormalize: bool = True,
):
    """Build a fused TopK gating softmax kernel.

    Args:
        num_experts: Number of MoE experts (columns in gating_output).
        topk: Number of top experts to select per token.
        dtype_str: Input data type ('f32', 'f16', 'bf16').
        renormalize: If True, rescale selected weights to sum to 1.

    Returns:
        A @flyc.jit launcher function.
    """
    arch = get_hip_arch()

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    # Shared memory layout:
    #   s_red_val:  RED_SLOTS f32  – for max / sum reductions
    #   s_red_val2: RED_SLOTS f32  – for argmax value in top-K
    #   s_red_idx:  RED_SLOTS i32  – for argmax index in top-K
    #   s_topk_out: topk f32       – selected weights (thread 0 accumulates)
    #   s_winner:   1 i32          – broadcast winning expert index
    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    i32_bytes = 4

    red_val_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_val_offset + RED_SLOTS * f32_bytes

    red_val2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_val2_offset + RED_SLOTS * f32_bytes

    red_idx_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_idx_offset + RED_SLOTS * i32_bytes

    topk_out_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = topk_out_offset + topk * f32_bytes

    winner_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = winner_offset + 1 * i32_bytes

    @flyc.kernel
    def topk_gating_softmax_kernel(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIndices: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast

        base_ptr = allocator.get_base()
        s_red_val = SmemPtr(base_ptr, red_val_offset, T.f32, shape=(RED_SLOTS,))
        s_red_val2 = SmemPtr(base_ptr, red_val2_offset, T.f32, shape=(RED_SLOTS,))
        s_red_idx = SmemPtr(base_ptr, red_idx_offset, T.i32, shape=(RED_SLOTS,))
        s_topk_out = SmemPtr(base_ptr, topk_out_offset, T.f32, shape=(topk,))
        s_winner = SmemPtr(base_ptr, winner_offset, T.i32, shape=(1,))
        s_red_val.get()
        s_red_val2.get()
        s_red_idx.get()
        s_topk_out.get()
        s_winner.get()

        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_log2e = arith.constant(1.4426950408889634, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)

        # ── wave / block reductions ──────────────────────────────────────
        def wave_reduce(x, mode):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                if mode == "max":
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce(val, mode):
            if RED_SLOTS == 1:
                return wave_reduce(val, mode)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            neutral = c_neg_inf if mode == "max" else c_zero_f

            w = wave_reduce(val, mode)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                s_red_val.store(w, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v = s_red_val.load([lane_safe_idx])
                z = neutral
                ww = in_range.select(v, z)
                ww = wave_reduce(ww, mode)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red_val.store(ww, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red_val.load([c0_idx])

        def wave_reduce_argmax(val, idx):
            """Butterfly argmax within a wave, returning (max_val, max_idx)."""
            width_i32 = fx.Int32(WARP_SIZE)
            wv = val
            wi = idx
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer_v = wv.shuffle_xor(off, width_i32)
                peer_i = wi.shuffle_xor(off, width_i32)
                is_greater = peer_v > wv
                is_equal = ArithValue(peer_v) == ArithValue(wv)
                peer_lower_idx = peer_i < wi
                take_peer = is_greater | (is_equal & peer_lower_idx)
                wv = take_peer.select(peer_v, wv)
                wi = take_peer.select(peer_i, wi)
            return wv, wi

        def block_reduce_argmax(val, idx):
            """Block-level argmax using shared memory, returns (val, idx) visible to all."""
            if RED_SLOTS == 1:
                wv, wi = wave_reduce_argmax(val, idx)
                if tid == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red_val2.store(wv, [c0_idx])
                    s_red_idx.store(wi, [c0_idx])
                gpu.barrier()
                c0_idx = fx.Index(0)
                return s_red_val2.load([c0_idx]), s_red_idx.load([c0_idx])

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            wv, wi = wave_reduce_argmax(val, idx)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                s_red_val2.store(wv, [wave_idx])
                s_red_idx.store(wi, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v = s_red_val2.load([lane_safe_idx])
                i = s_red_idx.load([lane_safe_idx])
                z_v = c_neg_inf
                z_i = fx.Int32(0)
                vv = in_range.select(v, z_v)
                ii = in_range.select(i, z_i)
                vv, ii = wave_reduce_argmax(vv, ii)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red_val2.store(vv, [c0_idx])
                    s_red_idx.store(ii, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red_val2.load([c0_idx]), s_red_idx.load([c0_idx])

        # ── scalar memory helpers ─────────────────────────────────────────
        GatingOutput_buf = fx.rocdl.make_buffer_tensor(GatingOutput)
        TopkWeights_buf = fx.rocdl.make_buffer_tensor(TopkWeights)
        TopkIndices_buf = fx.rocdl.make_buffer_tensor(TopkIndices)
        TokenExpertIndices_buf = fx.rocdl.make_buffer_tensor(TokenExpertIndices)

        row_gating = fx.slice(GatingOutput_buf, (bid, None))
        row_weights = fx.slice(TopkWeights_buf, (bid, None))
        row_indices = fx.slice(TopkIndices_buf, (bid, None))
        row_token_expert = fx.slice(TokenExpertIndices_buf, (bid, None))

        # gating input: elem_type scalars
        copy_atom_in = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        scalar_reg_ty_in = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        scalar_reg_lay = fx.make_layout(1, 1)

        # output: f32 for weights
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        scalar_reg_ty_f32 = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )

        # output: i32 for indices
        copy_atom_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        scalar_reg_ty_i32 = fx.MemRefType.get(
            T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )

        gating_div = fx.logical_divide(row_gating, fx.make_layout(1, 1))
        weights_div = fx.logical_divide(row_weights, fx.make_layout(1, 1))
        indices_div = fx.logical_divide(row_indices, fx.make_layout(1, 1))
        token_expert_div = fx.logical_divide(row_token_expert, fx.make_layout(1, 1))

        def _load_scalar_in(divided, index):
            view = fx.slice(divided, (None, index))
            r = fx.memref_alloca(scalar_reg_ty_in, scalar_reg_lay)
            fx.copy_atom_call(copy_atom_in, view, r)
            v = fx.memref_load_vec(r)
            return vector.extract(v, static_position=[0])

        def _store_scalar_f32(divided, index, val):
            r = fx.memref_alloca(scalar_reg_ty_f32, scalar_reg_lay)
            vec_ty = T.vec(1, T.f32)
            v = vector.from_elements(vec_ty, [val])
            fx.memref_store_vec(v, r)
            view = fx.slice(divided, (None, index))
            fx.copy_atom_call(copy_atom_f32, r, view)

        def _store_scalar_i32(divided, index, val):
            r = fx.memref_alloca(scalar_reg_ty_i32, scalar_reg_lay)
            vec_ty = T.vec(1, T.i32)
            v = vector.from_elements(vec_ty, [val])
            fx.memref_store_vec(v, r)
            view = fx.slice(divided, (None, index))
            fx.copy_atom_call(copy_atom_i32, r, view)

        # ==================================================================
        # Pass 1: Load gating logits + compute max for softmax
        # ==================================================================
        row_buffer = []
        thread_max = c_neg_inf

        for base in range_constexpr(0, num_experts, BLOCK_THREADS):
            idx = tid + base
            c_E = Int32(num_experts)
            is_valid = idx < c_E
            idx_safe = is_valid.select(idx, Int32(0))
            val_e = _load_scalar_in(gating_div, idx_safe)
            val = val_e if dtype_str == "f32" else val_e.extf(compute_type)
            safe_val = is_valid.select(val, c_neg_inf)
            row_buffer.append((safe_val, is_valid))
            thread_max = thread_max.maximumf(safe_val)

        global_max = block_reduce(thread_max, "max")

        # ==================================================================
        # Pass 2: exp(x - max) and sum for softmax normalization
        # ==================================================================
        thread_sum = c_zero_f
        softmax_buffer = []
        for safe_val, is_valid in row_buffer:
            sub = safe_val - ArithValue(global_max)
            scaled = sub * c_log2e
            exp_val = scaled.exp2(fastmath=fm_fast)
            safe_exp = is_valid.select(exp_val, c_zero_f)
            thread_sum = thread_sum + safe_exp
            softmax_buffer.append((exp_val, is_valid))

        global_sum = block_reduce(thread_sum, "sum")
        inv_sum = c_one_f / ArithValue(global_sum)

        # ==================================================================
        # Pass 3: Normalize to get softmax probabilities (in registers)
        # ==================================================================
        prob_buffer = []
        for exp_val, is_valid in softmax_buffer:
            norm_val = ArithValue(exp_val) * inv_sum
            safe_prob = is_valid.select(norm_val, c_neg_inf)
            prob_buffer.append((safe_prob, is_valid))

        # ==================================================================
        # Pass 4: Iterative Top-K selection
        # ==================================================================
        # We track probabilities in prob_buffer for argmax, and use a
        # separate working copy for masking out winners.
        # Each thread also tracks which column indices it owns.
        selected_sum = c_zero_f

        for k_idx in range_constexpr(topk):
            # Thread-local argmax
            thread_best_val = c_neg_inf
            thread_best_idx = fx.Int32(-1)
            buf_pos = 0

            for base in range_constexpr(0, num_experts, BLOCK_THREADS):
                prob_val, is_valid = prob_buffer[buf_pos]
                col_idx = tid + base
                col_idx_i32 = col_idx

                is_better = prob_val > thread_best_val
                thread_best_val = is_better.select(prob_val, thread_best_val)
                thread_best_idx = is_better.select(col_idx_i32, thread_best_idx)
                buf_pos += 1

            # Block-level argmax
            global_best_val, global_best_idx = block_reduce_argmax(
                thread_best_val, thread_best_idx
            )

            # Thread 0 writes outputs for this k iteration
            if tid == fx.Int32(0):
                k_idx_ir = fx.Index(k_idx)
                s_topk_out.store(global_best_val, [k_idx_ir])

                _store_scalar_f32(weights_div, Int32(k_idx), global_best_val)
                _store_scalar_i32(indices_div, Int32(k_idx), global_best_idx)

                # token_expert_indices = k_idx * num_rows + token_row
                # num_rows is grid dim x (= num_tokens), bid is the token row
                num_rows_val = ArithValue(fx.gpu.grid_dim.x)
                k_offset = Int32(k_idx) * num_rows_val
                tei_val = k_offset + bid
                _store_scalar_i32(token_expert_div, Int32(k_idx), tei_val)

                # Broadcast winner index for masking
                c0_idx = fx.Index(0)
                s_winner.store(global_best_idx, [c0_idx])

                selected_sum = selected_sum + global_best_val
            gpu.barrier()

            # All threads read the winner to mask it out
            c0_idx = fx.Index(0)
            winner_idx = s_winner.load([c0_idx])

            # Mask out the winning expert in each thread's buffer
            buf_pos2 = 0
            for base in range_constexpr(0, num_experts, BLOCK_THREADS):
                col_idx = tid + base
                prob_val, is_valid = prob_buffer[buf_pos2]
                is_winner = ArithValue(col_idx) == ArithValue(winner_idx)
                masked_val = is_winner.select(c_neg_inf, prob_val)
                prob_buffer[buf_pos2] = (masked_val, is_valid)
                buf_pos2 += 1

            gpu.barrier()

        # ==================================================================
        # Pass 5: Optional renormalization
        # ==================================================================
        if renormalize:
            if tid == fx.Int32(0):
                c_eps = arith.constant(1e-20, type=compute_type)
                denom = selected_sum.maximumf(c_eps)
                for k_idx in range_constexpr(topk):
                    k_idx_ir = fx.Index(k_idx)
                    w_val = s_topk_out.load([k_idx_ir])
                    w_norm = ArithValue(w_val) / ArithValue(denom)
                    _store_scalar_f32(weights_div, Int32(k_idx), w_norm)

    # ── JIT host launcher ─────────────────────────────────────────────────
    @flyc.jit
    def launch_topk_gating_softmax(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIndices: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
        num_tokens_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = ArithValue(num_tokens_in).index_cast(T.index)
        launcher = topk_gating_softmax_kernel(
            GatingOutput, TopkWeights, TopkIndices, TokenExpertIndices
        )
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_gating_softmax
