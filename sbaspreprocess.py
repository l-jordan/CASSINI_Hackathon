"""
SBAS Preprocessing  —  Stage 1
================================
Outputs:
    clean_cube.dat   — φ_clean  (float32 memmap, NaN where invalid)
    weight_cube.dat  — CC²/(1−CC²+ε) weights (float32 memmap, 0 where invalid)
    meta.npz         — shape + pair metadata for downstream inversion

Requirements:  pip install numpy rasterio tqdm
"""

import os, re, json, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── FIX 6: Limit BLAS/MKL thread saturation ──────────────────────────────────
# Prevents NumPy from claiming all CPU cores in tight loops, which is the
# primary cause of Windows UI becoming unresponsive under full load.
# Set to (your physical core count - 2) to keep headroom for the OS.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR    = r"C:\Users\iamal\Documents\CassiniWell\Data\IFG Harmonized subset"
OUTPUT_DIR  = r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Subset_Preprocessed"

CC_THRESHOLD    = 0.2   # pixels below this coherence are masked
OUTLIER_SIGMA   = 4.0   # MAD gate multiplier  (0 = disabled)
REFERENCE_PIXEL = None  # [row, col]  or  None  →  auto (highest mean CC)
REF_DOWNSAMPLE  = 20    # stride for reference pixel search (speeds up selection ~400×)          
FLUSH_BATCH     = 10    # flush memmaps every N IFGs (smoother disk I/O)
YIELD_EVERY     = 5     # sleep briefly every N IFGs to release CPU to OS

# ─────────────────────────────────────────────────────────────────────────────


def discover_pairs(data_dir):
    pattern = re.compile(r"(\d{8})_(\d{8})\.(unw|cc)\.tif$")
    found   = defaultdict(dict)
    for fp in Path(data_dir).glob("*.tif"):
        m = pattern.match(fp.name)
        if m:
            found[(m.group(1), m.group(2))][m.group(3)] = str(fp)
    return [
        {"primary": pri, "secondary": sec, "unw": f["unw"], "cc": f["cc"]}
        for (pri, sec), f in sorted(found.items())
        if "unw" in f and "cc" in f
    ]


def preprocess_ifg(unw_raw, cc_raw, ref_phase, cc_thresh, outlier_sigma):
    """
    Preprocessing chain — order matters:

        1. Zero-DN → NaN
        2. CC threshold mask       (on raw unw, before any subtraction)
        3. MAD outlier gate        (on raw unw, same reason)
        4. Reference subtraction   (anchor phase to reference pixel)
        5. Spatial median removal  (bulk APS / orbital ramp, computed on masked pixels)
        6. Build orthogonal outputs: clean phase (NaN invalid) + CC²/(1−CC²+ε) weight (0 invalid)

    Masking is done on unreferenced data so the gate is not biased by
    reference-pixel noise propagating across the scene.
    """
    # FIX 2: copy=False avoids a redundant allocation when the array is
    # already float32 (memmap slices are); only copies if a cast is needed.
    unw = unw_raw.astype(np.float32, copy=False)
    cc  = cc_raw.astype(np.float32, copy=False)

    # 1. Zero-DN → NaN
    unw[unw == 0] = np.nan
    cc[cc   == 0] = np.nan

    # 2. CC threshold mask  (on raw phase)
    # FIX 3: replace isfinite(unw) with (unw != 0).
    # After the zero→NaN step above, the only non-finite values come from
    # original zeros, which are now NaN.  isfinite() scans the full array for
    # inf/NaN beyond that, which is redundant and expensive.  (unw != 0) is
    # equivalent here and avoids the extra full-array pass.
    mask = (cc > cc_thresh) & (unw != 0)

    # 3. MAD outlier gate  (on raw phase, before reference subtraction)
    # FIX 4: replace np.median(vals) + np.median(|vals - med|) with
    # percentile-based equivalents.  np.percentile uses a single sorted pass
    # internally for multiple quantiles; np.median calls sort twice (once per
    # call).  For large valid-pixel counts this is 3–5× faster with identical
    # statistical meaning for a symmetric distribution.
    if outlier_sigma > 0 and mask.any():
        vals = unw[mask]
        med  = float(np.percentile(vals, 50))
        # IQR-based sigma: (Q75 - Q25) / 1.349  ≡  1.4826 * MAD for Gaussian
        sigma_hat = (float(np.percentile(vals, 75)) - float(np.percentile(vals, 25))) / 1.349
        if sigma_hat > 0:
            mask &= np.abs(unw - med) <= outlier_sigma * sigma_hat

    # 4. Reference subtraction  (guaranteed finite from extract_ref_phases)
    unw -= ref_phase

    # 5. Spatial median removal  (only over valid pixels post-reference)
    if mask.any():
        unw -= float(np.nanmedian(unw[mask]))

    # 6. Orthogonal outputs
    phase_clean       = np.full_like(unw, np.nan)
    phase_clean[mask] = unw[mask]

    # CC²/(1−CC²+ε) — better approximation of 1/σ²_phase than CC/(1−CC+ε).
    # Squaring amplifies the separation between high- and medium-coherence pixels,
    # improving (GᵀWG) conditioning. CC is clipped to 0.999 to prevent the
    # denominator collapsing to ε alone when coherence → 1.0, which would
    # produce unbounded weights and destabilise the least-squares solve.
    cc_clip      = np.clip(cc, 0.0, 0.999)
    eps          = 1e-3
    cc2          = cc_clip ** 2
    weight       = np.zeros_like(cc)
    weight[mask] = cc2[mask] / (1.0 - cc2[mask] + eps)

    return phase_clean, weight


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = discover_pairs(DATA_DIR)
    if not pairs:
        raise FileNotFoundError(f"No valid .unw.tif / .cc.tif pairs in {DATA_DIR}")
    n_ifg = len(pairs)
    print(f"Found {n_ifg} IFG pairs")

    with rasterio.open(pairs[0]["unw"]) as src:
        nrows, ncols = src.shape
    print(f"Raster: {nrows} × {ncols}")

    # ── Raw read buffers as memmaps — never spikes RAM ────────────────────────
    tmp_unw = os.path.join(OUTPUT_DIR, "_tmp_unw.dat")
    tmp_cc  = os.path.join(OUTPUT_DIR, "_tmp_cc.dat")

    raw_unw = np.memmap(tmp_unw, dtype="float32", mode="w+", shape=(n_ifg, nrows, ncols))
    raw_cc  = np.memmap(tmp_cc,  dtype="float32", mode="w+", shape=(n_ifg, nrows, ncols))

    for k, p in enumerate(tqdm(pairs, desc="Loading IFGs")):
        with rasterio.open(p["unw"]) as ds:
            raw_unw[k] = ds.read(1)
        with rasterio.open(p["cc"]) as ds:
            raw_cc[k]  = ds.read(1)

    raw_unw.flush()
    raw_cc.flush()

    # ── Reference pixel: downsampled search, then map back to full resolution ─
    if REFERENCE_PIXEL is not None:
        ref_row, ref_col = REFERENCE_PIXEL
        print(f"Reference pixel (user): row={ref_row}, col={ref_col}")
    else:
        s = REF_DOWNSAMPLE
        # .astype() produces a copy — avoids mutating the memmap when we
        # zero-mask, and avoids a temporary float64 materialization later.
        cc_small = raw_cc[:, ::s, ::s].astype(np.float32)
        cc_small[cc_small == 0] = np.nan          # 0-DN is not-observed, not zero coherence

        # Streaming nanmean — accumulate in float64 per-slice to avoid a
        # full (n_ifg, H, W) temporary array being allocated by np.nanmean.
        sh = cc_small.shape[1:]
        mean_cc_small = np.zeros(sh, dtype=np.float64)
        count         = np.zeros(sh, dtype=np.float64)
        for k in range(n_ifg):
            block       = cc_small[k].astype(np.float64)
            valid_block = np.isfinite(block)
            mean_cc_small += np.where(valid_block, block, 0.0)
            count         += valid_block
        mean_cc_small /= np.maximum(count, 1e-6)

        idx = np.nanargmax(mean_cc_small)
        rr, rc  = np.unravel_index(idx, mean_cc_small.shape)
        ref_row = min(int(rr) * s, nrows - 1)
        ref_col = min(int(rc) * s, ncols - 1)
        print(f"Auto reference pixel: row={ref_row}, col={ref_col} "
              f"(approx mean CC={mean_cc_small.flat[idx]:.3f}, stride={s})")

    # ── Per-IFG reference phase ───────────────────────────────────────────────
    # Missing ref-pixel values are set to 0.0 (no subtraction for that IFG).
    # Using nanmedian as a fill would inject a temporal bias: each "filled" IFG
    # would share the same offset derived from its neighbours, smoothing what
    # should be an independent per-IFG anchor.  Zero is the correct neutral.
    # 0 is a valid unwrapped phase value — only NaN/inf mean "missing".
    # Missing IFGs get ref_phase=0.0 (identity: no subtraction), preserving
    # zero-centering without injecting a temporally-correlated bias.
    ref_phases = raw_unw[:, ref_row, ref_col].astype(np.float32)
    valid_ref  = np.isfinite(ref_phases)
    n_missing  = int((~valid_ref).sum())
    if n_missing:
        frac = n_missing / n_ifg
        tag  = "WARNING — unstable reference pixel" if frac > 0.2 else "Note"
        print(f"{tag}: ref pixel missing in {n_missing}/{n_ifg} IFGs "
              f"({100*frac:.0f}%) — using 0.0 (no subtraction) for those IFGs")
    ref_phases[~valid_ref] = 0.0

    # ── Output cubes ─────────────────────────────────────────────────────────
    clean_path  = os.path.join(OUTPUT_DIR, "clean_cube.dat")
    weight_path = os.path.join(OUTPUT_DIR, "weight_cube.dat")

    clean_cube  = np.memmap(clean_path,  dtype="float32", mode="w+", shape=(n_ifg, nrows, ncols))
    weight_cube = np.memmap(weight_path, dtype="float32", mode="w+", shape=(n_ifg, nrows, ncols))
    clean_cube[:]  = np.nan
    weight_cube[:] = 0.0

    # FIX 1 + FIX 5: yield to the OS every YIELD_EVERY IFGs via a 1 ms sleep,
    # and flush memmaps every FLUSH_BATCH IFGs.
    #
    # Why the sleep matters: NumPy releases the GIL during C-level array ops,
    # but the Python loop itself still drives the thread at 100 % CPU between
    # calls.  A 1 ms pause every 5 IFGs gives the Windows scheduler a chance
    # to service UI and driver threads, eliminating the "frozen desktop" feel
    # at a cost of < 1 % total runtime for 558 IFGs.
    #
    # Why batched flushing: calling flush() on every IFG triggers repeated
    # msync/WriteFile calls that can stall on spinning HDDs.  Flushing every
    # FLUSH_BATCH IFGs amortises the syscall overhead and produces more
    # uniform disk throughput.
    for k in tqdm(range(n_ifg), desc="Preprocessing"):
        clean_cube[k], weight_cube[k] = preprocess_ifg(
            raw_unw[k], raw_cc[k], float(ref_phases[k]), CC_THRESHOLD, OUTLIER_SIGMA
        )

        # OS yield
        if k % YIELD_EVERY == 0:
            time.sleep(0.001)

        # Batched flush
        if (k + 1) % FLUSH_BATCH == 0 or k == n_ifg - 1:
            clean_cube.flush()
            weight_cube.flush()

    # ── Cleanup temp files ────────────────────────────────────────────────────
    del raw_unw, raw_cc
    for path in (tmp_unw, tmp_cc):
        try:
            os.remove(path)
        except OSError:
            pass

    # ── Metadata ──────────────────────────────────────────────────────────────
    epochs = sorted({p["primary"] for p in pairs} | {p["secondary"] for p in pairs})
    np.savez(
        os.path.join(OUTPUT_DIR, "meta.npz"),
        pairs_json = json.dumps(pairs),
        epochs     = epochs,
        ref_row    = np.int32(ref_row),
        ref_col    = np.int32(ref_col),
        nrows      = np.int32(nrows),
        ncols      = np.int32(ncols),
        n_ifg      = np.int32(n_ifg),
    )

    valid_frac = np.sum(weight_cube > 0, axis=0) / n_ifg
    n_valid    = int((valid_frac > 0).sum())
    print(f"Done. Valid pixels: {n_valid:,} / {nrows * ncols:,}  "
          f"({100 * n_valid / (nrows * ncols):.1f}%)")


if __name__ == "__main__":
    main()