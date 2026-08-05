# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""PyTorch-facing wrapper for the FlyDSL KDA chunkwise forward kernel.

Accepts the usual ``[B, H, T, D]`` layout, reshapes into the packed per-chunk
layout the kernel expects, and caches one compiled module per configuration.
"""

import functools
import math

import torch

from kernels.linear_attn.kda_kernel import (
    fwd_solve,
    LDS_GRANULE,
    LDS_PER_CU,
    build_kda_fwd_module,
    fwd_lds_bytes,
)
from kernels.linear_attn.kda_split import build_kda_prep_module, build_kda_scan_module

# C=32 with a 4-way value-channel split keeps LDS at ~71 KB, so two workgroups fit
# per CU (160 KB).  That occupancy, plus the O(C^2) shrink of the sequential
# triangular solve, makes it ~2.9x faster than C=64 despite the extra chunks.
CHUNK_SIZE = 32
DV_SPLIT = 4

# The sequence-parallel path moves every state-independent tile out of the serial
# chunk walk, cutting it from ~8.8 to ~2.5 us per chunk, so it wins by 1.5-3x once
# there are enough chunks to amortize the extra kernel launch and the global round
# trip.  The crossover is on chunks *per head*, not batch: below ~32 the fused
# kernel's whole runtime is under the split's fixed overhead.  Measured, not
# derived; see bench_kda_split.py.
SPLIT_MIN_CHUNKS = 32

# The prep kernel materializes six tiles for every chunk, which is O(T) extra memory
# where the fused kernel needs none -- ~900 MB at B=32, H=16, T=2048.  Past a share
# of free memory that is not a good trade for 1.6x, so fall back rather than risk
# an OOM in a training step.
SPLIT_MEM_FRACTION = 0.25


def _split_workspace_bytes(BH, NC, C, DK):
    n = BH * NC
    return 2 * n * (2 * C * C + 2 * C * DK + DK * C) + 4 * n * DK


@functools.lru_cache(maxsize=8)
def _num_cus(device_index):
    """CUs on the target device.  256 on this gfx950 part, 304 on an MI300X -- read it,
    never hard-code it, or the dispatch rule below silently mistunes on other cards."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _wgs_per_cu(DK, DV, C, dv_split):
    """Resident workgroups per CU for a candidate split.

    LDS is handed out in ``LDS_GRANULE`` chunks, so the granule-rounded footprint --
    not the raw struct size -- is what divides into the 160 KB a CU has.
    """
    lds = fwd_lds_bytes(DK=DK, DV=DV, C=C, DV_SPLIT=dv_split)
    return LDS_PER_CU // (-(-lds // LDS_GRANULE) * LDS_GRANULE)


@functools.lru_cache(maxsize=256)
def _auto_dv_split(BH, DK, DV, C, num_cus, ev_max):
    """Value-channel split, chosen from the launch shape rather than from a constant.

    Two forces pull in opposite directions, and both are read off the device rather
    than hard-coded:

    * Splitting ``d`` ways multiplies the workgroup count by ``d``, but every one of
      those workgroups replicates the *whole* state-independent key side (T', A, the
      two Gamma tiles) and re-fetches q/k/g from HBM instead of sharing them through
      L2 -- 2.15x read amplification at d=4.  So a split past the point where the
      device still has idle CUs is pure waste: ``d_fill = max{d : BH*d <= NUM_CUS}``.
    * A workgroup is 4 waves on 4 SIMDs, so at one workgroup per CU the kernel runs at
      *one wave per SIMD* and its ~56 %-of-cycles wait time is hidden by nothing.  A
      coarser split than the finest one that still fits two workgroups per CU is
      therefore never worth taking: ``d_occ = min{d : WGs per CU >= 2}``.

    ``d = max(d_fill, d_occ)``, clamped to the legal splits.  Measured ``stream_ms``
    on gfx950 / 256 CU with this kernel's footprint (WGs/CU: d1 1, d2 2, d4 2):

        BH=8    d1 .1955  d2 .1355  [d4 .1250]      d_fill 4, d_occ 2 -> 4
        BH=64   d1 .2125  d2 .1426  [d4 .1313]      d_fill 4, d_occ 2 -> 4
        BH=128  d1 .2017 [d2 .1475]  d4 .1617       d_fill 2, d_occ 2 -> 2
        BH=256  d1 .4213 [d2 .3672]  d4 .6395       d_fill 1, d_occ 2 -> 2
        BH=512  d1 .4252 [d2 .3677]  d4 .6317       d_fill 1, d_occ 2 -> 2

    The ``d_occ`` clause only became reachable once the kernel's footprint at EV=64
    dropped under 80 KB; before that d2 was 88.7 KB (one workgroup per CU) and the
    big shapes measured 0.65 ms at d2 against 0.43 ms at d1.
    """
    legal = [
        d
        for d in (1, 2, 4)
        if DV % d == 0
        and DV // d >= 32  # EV is an MMA M/N extent the 2x2 wave grid halves
        and DV // d <= ev_max
        and fwd_lds_bytes(DK=DK, DV=DV, C=C, DV_SPLIT=d) <= LDS_PER_CU
    ]
    if not legal:
        return DV_SPLIT
    fits = [d for d in legal if BH * d <= num_cus]
    d_fill = max(fits) if fits else min(legal)
    occ2 = [d for d in legal if _wgs_per_cu(DK, DV, C, d) >= 2]
    d_occ = min(occ2) if occ2 else d_fill
    return min(max(legal), max(d_fill, d_occ))


@functools.lru_cache(maxsize=64)
def _get_module(BH, T, DK, DV, C, DV_SPLIT, BLOCK, has_h0, out_dtype_str, debug_stage="", solve="blocked"):
    out_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[out_dtype_str]
    import flydsl.expr as fx

    fx_out = {"bf16": fx.BFloat16, "fp32": fx.Float32}[out_dtype_str]
    mod = build_kda_fwd_module(
        BH=BH,
        T=T,
        DK=DK,
        DV=DV,
        C=C,
        DV_SPLIT=DV_SPLIT,
        BLOCK=BLOCK,
        has_initial_state=has_h0,
        store_final_state=True,
        out_dtype=fx_out,
        debug_stage=debug_stage,
        solve=solve,
    )
    return mod, out_dtype


@functools.lru_cache(maxsize=64)
def _get_split_modules(BH, T, DK, DV, C, DV_SPLIT, BLOCK, has_h0, out_dtype_str):
    out_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[out_dtype_str]
    import flydsl.expr as fx

    fx_out = {"bf16": fx.BFloat16, "fp32": fx.Float32}[out_dtype_str]
    prep = build_kda_prep_module(BH=BH, T=T, DK=DK, DV=DV, C=C, BLOCK=BLOCK)
    scan = build_kda_scan_module(
        BH=BH,
        T=T,
        DK=DK,
        DV=DV,
        C=C,
        DV_SPLIT=DV_SPLIT,
        BLOCK=BLOCK,
        has_initial_state=has_h0,
        store_final_state=True,
        out_dtype=fx_out,
    )
    return prep, scan, out_dtype


def _split_workspace(BH, NC, C, DK, DV, device):
    """Six per-chunk tiles the prep kernel hands to the scan kernel.

    Carved out of two allocations rather than six: at short sequences the host-side
    cost of the extra ``torch.empty`` calls is a visible part of the split's fixed
    overhead, which is what sets the dispatch crossover.
    """
    n = BH * NC
    shapes = [
        ("a", (n, C, C)),
        ("gk", (n, C, DK)),
        ("gq", (n, C, DK)),
        ("aqk", (n, C, C)),
        ("kt", (n, DK, C)),
    ]
    counts = [math.prod(shape) for _, shape in shapes]
    pool = torch.empty(sum(counts), dtype=torch.bfloat16, device=device)
    ws, off = {}, 0
    for (name, shape), cnt in zip(shapes, counts):
        ws[name] = pool[off : off + cnt].view(*shape)
        off += cnt
    ws["dec"] = torch.empty(n, DK, dtype=torch.float32, device=device)
    ws["_pool"] = pool  # keep the backing storage alive
    return ws


# ---------------------------------------------------------------------------
# Fast host dispatch.
#
# ``JitFunction.__call__`` re-derives its compilation cache key on *every* call
# (``inspect.Signature.bind`` + a per-argument ``__cache_signature__`` + a
# globals-drift snapshot): ~66 us of pure Python per launch, which is more than
# the GPU time of the small shapes and therefore the hard floor of the whole op.
#
# ``flyc.compile(jit_fn, *args)`` returns a ``CompiledFunction`` that resolves all
# of that once and then goes straight to the pre-built ctypes ``CallState``
# (~8 us).  Constexprs are baked at compile time, so the compiled callable is only
# valid for the configuration it was built with -- it is therefore cached under the
# *full* module config key, exactly like the module itself, and every argument
# (including the stream) is passed positionally.
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}

# FlyDSL's stream packer takes a raw ``hipStream_t`` int directly, so skip building
# a throwaway ``torch.cuda.Stream`` wrapper (~2.3 us) on every launch.  This still
# reads the *live* current stream, so stream-context semantics are unchanged.
_raw_stream = torch._C._cuda_getCurrentRawStream

# cfg tuple -> (compiled_launcher, torch_out_dtype).  One flat dict lookup on the
# hot path instead of an lru_cache hash of a 10-tuple plus a module unpack.
_FAST_LAUNCH = {}


def _build_fast_launch(key, args, stream_obj):
    """Compile ``launch_kda_fwd`` for this config and memoize the fast callable.

    ``flyc.compile`` *executes* the kernel once with ``args`` (that is how it
    obtains the ``CallState``), so the caller must NOT launch again: the outputs
    in ``args`` are already valid.

    Measured dead end, do not re-try: handing the tensors over as *static* memrefs
    (``flyc.from_torch_tensor``) does cut host dispatch further (17.8 -> 14.7 us,
    the layout struct-pack disappears) but the baked-in shapes make the kernel
    itself 0.3-0.7 % slower on every shape -- net 1.038x vs 1.044x.  Layout-dynamic
    memrefs win.
    """
    import flydsl.compiler as flyc

    mod, torch_out_dtype = _get_module(*key)
    compiled = flyc.compile(mod, *args, stream_obj)
    entry = (compiled, torch_out_dtype)
    _FAST_LAUNCH[key] = entry
    return entry


def kda_chunk_fwd(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    scale=None,
    chunk_size=CHUNK_SIZE,
    dv_split=None,
    block=256,
    out_dtype="bf16",
    output_final_state=True,
    stream=None,
    debug_stage="",
    split=None,
):
    """Chunkwise KDA forward.

    Parameters
    ----------
    q, k : [B, H, T, DK] bf16
    v    : [B, H, T, DV] bf16
    g    : [B, H, T, DK] fp32, per-channel log decay (<= 0), NOT pre-summed
    beta : [B, H, T]     fp32
    initial_state : [B, H, DK, DV] fp32 or None
    scale : q scaling, defaults to DK ** -0.5
    split : force the sequence-parallel two-kernel path on/off.  ``None`` picks it
        whenever the fused path would leave the GPU under-occupied.

    Returns
    -------
    o  : [B, H, T, DV]
    ht : [B, H, DK, DV] fp32 final state (or None)
    """
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert T % C == 0, f"T={T} must be a multiple of {C}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and g.is_contiguous()
    scale = DK**-0.5 if scale is None else float(scale)

    BH, NC = B * H, T // C
    if dv_split is None:
        # ev_max: the kernel stages an initial state through the (C x DK) g_cum tile,
        # so EV has to stay <= C on that path.
        dv_split = _auto_dv_split(
            BH, DK, DV, C, _num_cus(q.device.index), C if initial_state is not None else DV
        )
    EV = DV // dv_split
    if split is None:
        split = NC >= SPLIT_MIN_CHUNKS and not debug_stage
        if split:
            free, _ = torch.cuda.mem_get_info(q.device)
            split = _split_workspace_bytes(BH, NC, C, DK) < SPLIT_MEM_FRACTION * free
    if split and debug_stage:
        raise ValueError("debug_stage taps are only implemented for the fused path")

    # ``view`` rather than ``reshape``: contiguity is already asserted above, so the
    # copy fallback can never fire and this skips reshape's re-derivation of it.
    n2 = BH * NC
    q2 = q.view(n2, C * DK)
    k2 = k.view(n2, C * DK)
    g2 = g.view(n2, C * DK)
    b2 = beta.view(n2, C)
    v4 = v.view(n2, C, dv_split, EV)

    has_h0 = initial_state is not None
    if split:
        prep, scan, torch_out_dtype = _get_split_modules(
            BH, T, DK, DV, C, dv_split, block, has_h0, out_dtype
        )
    else:
        key = (
            BH, T, DK, DV, C, dv_split, block, has_h0, out_dtype, debug_stage,
            fwd_solve(BH, dv_split, _num_cus(q.device.index), C, block),
        )
        entry = _FAST_LAUNCH.get(key)
        torch_out_dtype = entry[1] if entry is not None else _TORCH_DTYPE[out_dtype]

    dev = q.device
    o = torch.empty(BH * NC, C, dv_split, EV, dtype=torch_out_dtype, device=dev)
    # the kernel keeps the state transposed (S^T, dv x dk)
    htt = torch.empty(BH, dv_split, EV, DK, dtype=torch.float32, device=dev)
    if has_h0:
        h0t = initial_state.transpose(-1, -2).contiguous().reshape(BH, dv_split, EV, DK)
    else:
        h0t = htt  # unread when the module was built with has_initial_state=False

    if split:
        if stream is None:
            stream = torch.cuda.current_stream()
        ws = _split_workspace(BH, NC, C, DK, DV, q.device)
        prep(
            q2, k2, g2, b2,
            ws["a"], ws["gk"], ws["gq"], ws["aqk"], ws["kt"], ws["dec"],
            float(scale), stream=stream,
        )
        # the scan stages these with 128-bit reads, so hand it flat views
        flat = lambda x: x.reshape(BH * NC, -1)
        scan(
            flat(ws["a"]), flat(ws["gk"]), flat(ws["gq"]), flat(ws["aqk"]),
            flat(ws["kt"]), ws["dec"],
            v4, o, h0t, htt, NC, stream=stream,
        )
    elif entry is not None:
        # hot path: pre-resolved ctypes dispatch, everything positional.  A raw
        # ``hipStream_t`` int is what the packer wants anyway, and reading it costs
        # ~0.2 us against ~2.3 us for a fresh ``torch.cuda.Stream`` wrapper.
        entry[0](
            q2, k2, g2, b2, v4, o, h0t, htt, NC, scale,
            _raw_stream(dev.index) if stream is None else stream,
        )
    else:
        # cold path: compiling also *runs* the kernel with these exact arguments,
        # so the outputs below are already valid -- do not launch a second time.
        _build_fast_launch(
            key, (q2, k2, g2, b2, v4, o, h0t, htt, NC, float(scale)),
            torch.cuda.current_stream() if stream is None else stream,
        )

    if debug_stage:
        # arg_ht carries the requested intermediate of workgroup (bh=0, sp=0)
        return o.reshape(B, H, T, DV), htt[0, 0]

    # htt is [BH, dv_split, EV, DK] contiguous, so one view + one transpose lands on
    # exactly the tensor the old reshape/transpose/reshape chain produced (same
    # shape, same strides, same storage -- verified elementwise), one op cheaper.
    o = o.view(B, H, T, DV)
    ht = htt.view(B, H, DV, DK).transpose(-1, -2) if output_final_state else None
    return o, ht
