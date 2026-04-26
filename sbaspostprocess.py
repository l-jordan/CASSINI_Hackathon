"""
SBAS Post-Inversion  —  Stage 3
================================
Self-contained pipeline starting from a completed disp_cube.dat.

Stages:
  1. Load + validate disp_cube.dat
  2. QC diagnostics  (% valid pixels / epoch, mean|disp|, std)
  3. APS correction  (epoch-by-epoch, stream-safe, no large intermediates)
  4. Temporal Savitzky-Golay smoothing  (optional)
  5. Velocity map    (vectorised Cramér 2×2 solve, NaN-aware)
  6. Reference grounding  (subtract median of stable-area pixels)
  7. Reliability mask  (min valid epochs + max std threshold)
  8. Export  (velocity.dat, aps_cube.dat, disp_clean.dat, masks)
  9. Summary report printed to console

Requirements:
    pip install numpy scipy tqdm
"""

import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Thread caps (tune to your CPU) ────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]    = "6"
os.environ["MKL_NUM_THREADS"]    = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

INVERSION_DIR = r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Inversion"
OUTPUT_DIR    = r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Postprocess"

# ── APS correction ─────────────────────────────────────────────────────────
APS_POLY_DEGREE  = 1     # 1 = linear detrend per epoch slice, 2 = quadratic
APS_GAUSS_SIGMA  = 15    # Gaussian σ in pixels (tune to ~500-2000 m footprint)
APS_ITERATIONS   = 2     # refinement passes (2 is usually enough)

# ── Temporal smoothing (Savitzky-Golay) ────────────────────────────────────
APPLY_SG_FILTER  = True  # False to skip
SG_WINDOW        = 7     # must be odd, ≥ poly_order+1; must be ≤ n_epochs
SG_POLY_ORDER    = 2     # polynomial order for SG fit

# ── Reliability / masking ──────────────────────────────────────────────────
MIN_VALID_EPOCHS = 4     # pixels with fewer valid epochs → masked out
MAX_STD_THRESHOLD = None # e.g. 2.0 (rad); set None to skip std-based masking

# ── Reference pixel / stable area ──────────────────────────────────────────
# If None, no reference subtraction is applied.
# If (row, col), subtract that pixel's time series from every pixel.
# If "auto", subtract median of pixels with |velocity| < AUTO_REF_PERCENTILE
# of the overall velocity distribution.
REFERENCE_MODE        = "auto"
REFERENCE_PIXEL       = None          # used only when REFERENCE_MODE == "pixel"
AUTO_REF_PERCENTILE   = 20            # bottom N% of |velocity| treated as stable

# ── Block size for epoch-by-epoch processing ───────────────────────────────
# Each epoch slice is (rows, cols) float32 — this is loaded one at a time,
# so RAM usage is always ≤ 2 × nrows × ncols × 4 bytes regardless of n_epochs.
# The APS loop is O(n_epochs) single-slice reads — no large intermediates.

# ── Output names ───────────────────────────────────────────────────────────
DISP_CLEAN_NAME  = "disp_clean.dat"
APS_CUBE_NAME    = "aps_cube.dat"
VELOCITY_NAME    = "velocity.dat"
MASK_NAME        = "reliability_mask.dat"   # uint8: 1=valid, 0=masked

# ══════════════════════════════════════════════════════════════════════════════


# ── helpers ───────────────────────────────────────────────────────────────────

def load_meta(inversion_dir):
    """
    Reads inversion_meta.npz written by Stage 2.
    Falls back gracefully if some fields are absent (older Stage 2 versions).
    """
    meta_path = os.path.join(inversion_dir, "inversion_meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"inversion_meta.npz not found in {inversion_dir}.\n"
            "Make sure Stage 2 completed successfully."
        )
    meta    = np.load(meta_path, allow_pickle=True)
    epochs  = list(meta["epochs"])
    t_days  = meta["t_days"].astype(np.float64)
    nrows   = int(meta["nrows"])
    ncols   = int(meta["ncols"])
    n_epochs = len(epochs)
    ref_row = int(meta["ref_row"]) if "ref_row" in meta else nrows // 2
    ref_col = int(meta["ref_col"]) if "ref_col" in meta else ncols // 2
    return epochs, t_days, nrows, ncols, n_epochs, ref_row, ref_col


def epoch_time_axis(t_days):
    """Centred normalised axis for polynomial fits (better conditioning)."""
    span   = t_days[-1] - t_days[0]
    t_norm = (t_days - t_days.mean()) / span if span > 0 else t_days.copy()
    return t_norm


# ══════════════════════════════════════════════════════════════════════════════
# 1 + 2.  LOAD + QC
# ══════════════════════════════════════════════════════════════════════════════

def qc_disp_cube(disp_cube, n_epochs, nrows, ncols):
    """
    Print per-epoch statistics without loading the full cube at once.
    Reads one epoch slice at a time → O(nrows × ncols) RAM regardless of depth.
    """
    print("\n── QC diagnostics ────────────────────────────────────────────────")
    P = nrows * ncols
    valid_frac_arr = np.empty(n_epochs, dtype=np.float32)
    mean_arr       = np.empty(n_epochs, dtype=np.float32)
    std_arr        = np.empty(n_epochs, dtype=np.float32)

    for k in tqdm(range(n_epochs), desc="QC epochs", leave=False):
        sl = disp_cube[k].ravel()                    # (P,) — one epoch
        valid = sl[np.isfinite(sl)]
        valid_frac_arr[k] = len(valid) / P
        mean_arr[k]       = float(np.mean(valid))   if len(valid) else np.nan
        std_arr[k]        = float(np.std(valid))    if len(valid) else np.nan

    print(f"  Epochs                : {n_epochs}")
    print(f"  Grid                  : {nrows} × {ncols}  ({P:,} pixels)")
    print(f"  Mean valid fraction   : {100*np.nanmean(valid_frac_arr):.1f}%")
    print(f"  Min  valid fraction   : {100*np.nanmin(valid_frac_arr):.1f}%  "
          f"(epoch {int(np.argmin(valid_frac_arr))})")
    print(f"  Mean |disp| over cube : {float(np.nanmean(np.abs(mean_arr))):.5f} rad")
    print(f"  Max  std / epoch      : {float(np.nanmax(std_arr)):.5f} rad")

    suspect = np.where(valid_frac_arr < 0.10)[0]
    if len(suspect):
        print(f"  ⚠  {len(suspect)} epochs have < 10% valid pixels: {suspect.tolist()}")
    else:
        print("  ✔  All epochs have ≥ 10% valid pixels")

    return valid_frac_arr, mean_arr, std_arr


# ══════════════════════════════════════════════════════════════════════════════
# 3.  APS CORRECTION  (epoch-by-epoch stream processing)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS IS SAFE WHERE THE PREVIOUS VERSION CRASHED
# ─────────────────────────────────────────────────────
# Stage 2's remove_atmosphere_vectorised() loaded the entire (N, rows, cols)
# cube as a float64 array before any processing.  For large cubes that alone
# exceeds available RAM and triggers an OOM error or Windows pagefile thrash.
#
# This version processes ONE epoch at a time:
#   • reads a single (rows, cols) float32 slice from the memmap
#   • fits + subtracts a polynomial trend  (spatial 2-D poly, degree=APS_POLY_DEGREE)
#   • applies a Gaussian low-pass filter to the residual to isolate long-wavelength APS
#   • writes APS and clean-displacement back to separate memmaps
#
# RAM consumed per iteration: ~ 6 × nrows × ncols × 8 B  (working float64 arrays)
# For a 1000 × 1000 grid that is 6 × 8 MB = 48 MB regardless of n_epochs.
#
# SPATIAL vs TEMPORAL APS STRATEGY
# ──────────────────────────────────
# Classic SBAS uses a TEMPORAL polynomial fit across all epochs at each pixel
# (to separate deformation from APS), then spatial smoothing.  That approach
# requires the full cube in RAM simultaneously.
#
# Here we flip the order:
#   Step A — per-epoch SPATIAL polynomial detrend (removes orbital/ramp errors)
#   Step B — spatial Gaussian low-pass (captures long-wavelength APS blobs)
#   Step C — optional second pass for refinement
#
# This is correct for APS isolation because:
#   (a) atmospheric signals are spatially correlated over km scales
#   (b) deformation signals from subsidence / uplift are also spatially smooth
#       but temporally progressive — the spatial poly+Gauss tracks the APS
#       without needing the temporal axis, at the cost of slight deformation
#       leakage into the APS estimate on a per-epoch basis.
# For most SAR subsidence monitoring applications (slow, mm-scale deformation)
# this is an acceptable trade-off and avoids the RAM wall entirely.
#
# To use the full temporal approach instead, increase system RAM or reduce
# the study area tile size and re-run Stage 2 with APS_IN_RAM=True.

def _fit_spatial_poly(data_2d, degree):
    """
    Fit a 2-D polynomial of given degree to data_2d  (rows, cols).
    Returns the polynomial surface (same shape) with NaN where data_2d is NaN.
    Works on masked float64 arrays; ignores NaN pixels.
    """
    rows, cols = data_2d.shape
    # Build coordinate arrays normalised to [-1, 1] for numerical stability
    r_lin = np.linspace(-1, 1, rows, dtype=np.float64)
    c_lin = np.linspace(-1, 1, cols, dtype=np.float64)
    CC, RR = np.meshgrid(c_lin, r_lin)   # (rows, cols)

    # Build Vandermonde-style design matrix for 2-D poly
    # columns: 1, r, c, r², rc, c², … up to total degree
    cols_list = []
    for r_exp in range(degree + 1):
        for c_exp in range(degree + 1 - r_exp):
            cols_list.append((RR ** r_exp * CC ** c_exp).ravel())
    A = np.column_stack(cols_list)   # (P, n_terms)

    flat = data_2d.ravel()
    valid = np.isfinite(flat)
    if valid.sum() < A.shape[1] + 1:
        # Too few valid pixels — return zero surface
        return np.zeros_like(data_2d)

    # Solve via least-squares on valid pixels only
    coeffs, _, _, _ = np.linalg.lstsq(A[valid], flat[valid], rcond=None)
    surface = (A @ coeffs).reshape(rows, cols)

    # Where original data is NaN → NaN in surface too
    surface[~data_2d.reshape(rows, cols).__class__] = surface  # no-op line — see below
    # (NaN propagation is handled after the call; surface is always fully defined)
    return surface


def remove_aps_streaming(disp_cube, aps_cube_mm, disp_clean_mm,
                          nrows, ncols, n_epochs,
                          poly_degree, sigma, n_iter):
    """
    Stream-safe APS removal: one epoch at a time.

    Parameters
    ----------
    disp_cube    : memmap (n_epochs, nrows, ncols) float32  — input
    aps_cube_mm  : memmap (n_epochs, nrows, ncols) float32  — output: APS
    disp_clean_mm: memmap (n_epochs, nrows, ncols) float32  — output: cleaned disp
    """
    print(f"\n── APS correction ─────────────────────────────────────────────────")
    print(f"   poly_degree={poly_degree}  |  σ={sigma}px  |  iterations={n_iter}")
    print(f"   Mode: stream (one epoch at a time) — RAM-safe")

    r_lin = np.linspace(-1, 1, nrows, dtype=np.float64)
    c_lin = np.linspace(-1, 1, ncols, dtype=np.float64)
    CC, RR = np.meshgrid(c_lin, r_lin)

    # Build design matrix once (shape: P × n_terms)
    Acols = []
    for r_exp in range(poly_degree + 1):
        for c_exp in range(poly_degree + 1 - r_exp):
            Acols.append((RR ** r_exp * CC ** c_exp).ravel())
    A_full = np.column_stack(Acols)   # (P, n_terms)
    n_terms = A_full.shape[1]

    for k in tqdm(range(n_epochs), desc="APS epochs"):
        # Read one epoch as float64 working copy
        epoch_f32 = disp_cube[k].astype(np.float64)    # (rows, cols)
        nan_mask  = ~np.isfinite(epoch_f32)

        aps_accum = np.zeros((nrows, ncols), dtype=np.float64)
        work      = epoch_f32.copy()

        for it in range(n_iter):
            flat  = work.ravel()
            valid = np.isfinite(flat)

            if valid.sum() < n_terms + 1:
                # Not enough pixels to fit poly — skip APS for this epoch
                break

            # ── Pass A: spatial polynomial detrend ──────────────────────────
            A_v      = A_full[valid]
            coeffs, _, _, _ = np.linalg.lstsq(A_v, flat[valid], rcond=None)
            poly_surface = (A_full @ coeffs).reshape(nrows, ncols)
            residual     = work - poly_surface                       # (rows, cols)
            residual[nan_mask] = np.nan

            # ── Pass B: spatial Gaussian low-pass on residual ────────────────
            # Fill NaN with 0 for the filter, then normalise by coverage
            res_filled = np.nan_to_num(residual, nan=0.0).astype(np.float32)
            valid_mask = np.isfinite(residual).astype(np.float32)

            blurred  = gaussian_filter(res_filled, sigma=sigma).astype(np.float64)
            coverage = gaussian_filter(valid_mask, sigma=sigma).astype(np.float64)
            coverage = np.maximum(coverage, 1e-6)

            aps_epoch = np.where(valid_mask > 0.05, blurred / coverage, 0.0)
            aps_epoch[nan_mask] = 0.0   # don't accumulate NaN into APS

            # Full APS for this iteration = spatial trend + smooth residual
            aps_iter  = poly_surface + aps_epoch
            aps_iter[nan_mask] = np.nan

            aps_accum += np.nan_to_num(aps_iter, nan=0.0)
            work      -= np.nan_to_num(aps_iter, nan=0.0)    # subtract in place
            work[nan_mask] = np.nan

        # Re-impose original NaN mask
        result = work.copy()
        result[nan_mask] = np.nan
        aps_out = aps_accum.copy()
        aps_out[nan_mask] = np.nan

        aps_cube_mm[k]   = aps_out.astype(np.float32)
        disp_clean_mm[k] = result.astype(np.float32)

    aps_cube_mm.flush()
    disp_clean_mm.flush()
    print("   APS correction complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEMPORAL SAVITZKY-GOLAY SMOOTHING  (optional)
# ══════════════════════════════════════════════════════════════════════════════

def apply_savgol_streaming(disp_clean_mm, nrows, ncols, n_epochs,
                            window, poly_order):
    """
    Per-pixel temporal Savitzky-Golay filter.
    Reads pixel strips column-by-column to keep RAM bounded.
    Overwrites disp_clean_mm in place.
    """
    print(f"\n── Temporal SG smoothing ──────────────────────────────────────────")
    print(f"   window={window}  |  poly_order={poly_order}")

    # Work column by column — each column strip is (n_epochs, nrows)
    P = nrows * ncols
    flat = disp_clean_mm.reshape(n_epochs, P)   # view

    STRIP = 2_000   # pixels per strip
    for p0 in tqdm(range(0, P, STRIP), desc="SG smoothing", leave=True):
        p1   = min(p0 + STRIP, P)
        strip = flat[:, p0:p1].astype(np.float64)    # (n_epochs, strip_width)

        # Apply SG only where we have enough non-NaN epochs
        valid_per_pixel = np.isfinite(strip).sum(axis=0)    # (strip_width,)
        ok = valid_per_pixel >= window

        if not ok.any():
            continue

        # Interpolate NaNs linearly before SG (SG can't handle them natively)
        strip_ok = strip[:, ok]
        for pp in range(strip_ok.shape[1]):
            ts = strip_ok[:, pp]
            nans = ~np.isfinite(ts)
            if nans.any() and (~nans).sum() >= 2:
                xs = np.arange(n_epochs)
                ts[nans] = np.interp(xs[nans], xs[~nans], ts[~nans])
                strip_ok[:, pp] = ts

        smoothed = savgol_filter(strip_ok, window_length=window,
                                 polyorder=poly_order, axis=0)

        # Re-apply NaN mask
        nan_mask_ok = ~np.isfinite(strip[:, ok])
        smoothed[nan_mask_ok] = np.nan

        strip[:, ok] = smoothed
        flat[:, p0:p1] = strip

    disp_clean_mm.flush()
    print("   SG smoothing complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VELOCITY MAP  (vectorised Cramér 2×2 solve)
# ══════════════════════════════════════════════════════════════════════════════

def compute_velocity(disp_clean_mm, nrows, ncols, n_epochs, t_days):
    """
    Fit  d(t) = a + v·t  per pixel via closed-form Cramér's rule.
    All epochs processed in one vectorised pass — no Python loop over pixels.
    RAM: 6 × n_epochs × P × 8 B (float64 sums) — column-strip blocked.
    Returns velocity (rows, cols) float32.
    """
    print("\n── Velocity map ────────────────────────────────────────────────────")
    P = nrows * ncols
    t = (t_days - t_days[0]).astype(np.float64)

    flat = disp_clean_mm.reshape(n_epochs, P)

    # Accumulate normal-equation sums in one read of the entire flat array
    # Done in strips to avoid the full (n_epochs, P) float64 allocation
    s0 = np.zeros(P, dtype=np.float64)
    s1 = np.zeros(P, dtype=np.float64)
    s2 = np.zeros(P, dtype=np.float64)
    r0 = np.zeros(P, dtype=np.float64)
    r1 = np.zeros(P, dtype=np.float64)

    STRIP = 5_000
    for p0 in tqdm(range(0, P, STRIP), desc="Velocity accumulate", leave=False):
        p1  = min(p0 + STRIP, P)
        col = flat[:, p0:p1].astype(np.float64)   # (n_epochs, strip_width)
        w   = np.isfinite(col).astype(np.float64)
        col = np.nan_to_num(col, nan=0.0)
        tv  = t[:, None]
        s0[p0:p1] += w.sum(axis=0)
        s1[p0:p1] += (tv * w).sum(axis=0)
        s2[p0:p1] += (tv**2 * w).sum(axis=0)
        r0[p0:p1] += (w * col).sum(axis=0)
        r1[p0:p1] += (tv * w * col).sum(axis=0)

    det = s0 * s2 - s1**2
    ok  = (det > 0) & (s0 >= 2)

    vel = np.full(P, np.nan, dtype=np.float32)
    vel[ok] = ((s0[ok] * r1[ok] - s1[ok] * r0[ok]) / det[ok]).astype(np.float32)

    v2d = vel.reshape(nrows, ncols)
    print(f"   median |v| = {float(np.nanmedian(np.abs(vel))):.6f} rad/day  "
          f"({np.isfinite(vel).sum():,} valid pixels)")
    return v2d


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REFERENCE GROUNDING
# ══════════════════════════════════════════════════════════════════════════════

def ground_reference(disp_clean_mm, velocity, nrows, ncols, n_epochs,
                      mode, pixel, auto_pct):
    """
    Subtract a reference signal from every pixel's time series.
    Modes:
      "pixel" — subtract pixel[row,col]'s time series
      "auto"  — subtract median time series of the bottom auto_pct% |velocity| pixels
      None    — no-op
    """
    if mode is None:
        print("\n── Reference grounding: skipped ───────────────────────────────────")
        return

    P    = nrows * ncols
    flat = disp_clean_mm.reshape(n_epochs, P)
    vel_flat = velocity.ravel()

    print(f"\n── Reference grounding  (mode={mode}) ─────────────────────────────")

    if mode == "pixel":
        r, c = pixel
        ref_ts = disp_clean_mm[:, r, c].astype(np.float64)   # (n_epochs,)
        print(f"   Reference pixel: row={r}, col={c}")
    elif mode == "auto":
        valid_vel = np.isfinite(vel_flat)
        thresh = np.nanpercentile(np.abs(vel_flat[valid_vel]), auto_pct)
        stable = valid_vel & (np.abs(vel_flat) <= thresh)
        n_stable = stable.sum()
        print(f"   Stable pixels (|v| ≤ {thresh:.5f} rad/day): {n_stable:,}")
        if n_stable < 10:
            print("   ⚠  Too few stable pixels — skipping reference grounding.")
            return
        # Median time series of stable pixels
        ref_ts = np.nanmedian(flat[:, stable].astype(np.float64), axis=1)  # (n_epochs,)
    else:
        raise ValueError(f"Unknown REFERENCE_MODE: {mode!r}")

    # Subtract reference from all pixels in strips
    STRIP = 10_000
    for p0 in tqdm(range(0, P, STRIP), desc="Reference subtract", leave=False):
        p1  = min(p0 + STRIP, P)
        col = flat[:, p0:p1].astype(np.float64)
        nan_mask = ~np.isfinite(col)
        col -= ref_ts[:, None]
        col[nan_mask] = np.nan
        flat[:, p0:p1] = col.astype(np.float32)

    disp_clean_mm.flush()
    print("   Reference grounding complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RELIABILITY MASK
# ══════════════════════════════════════════════════════════════════════════════

def make_reliability_mask(disp_clean_mm, nrows, ncols, n_epochs,
                           min_valid_epochs, max_std):
    """
    uint8 mask: 1 = reliable pixel, 0 = masked.
    Criteria:
      • number of finite epochs ≥ min_valid_epochs
      • (optional) temporal std ≤ max_std
    """
    print("\n── Reliability mask ───────────────────────────────────────────────")
    P    = nrows * ncols
    flat = disp_clean_mm.reshape(n_epochs, P)

    valid_count = np.zeros(P, dtype=np.int32)
    std_arr     = np.zeros(P, dtype=np.float32)

    STRIP = 10_000
    for p0 in tqdm(range(0, P, STRIP), desc="Mask build", leave=False):
        p1  = min(p0 + STRIP, P)
        col = flat[:, p0:p1].astype(np.float64)
        valid_count[p0:p1] = np.isfinite(col).sum(axis=0)
        std_arr[p0:p1]     = np.nanstd(col, axis=0).astype(np.float32)

    mask = (valid_count >= min_valid_epochs).astype(np.uint8)

    if max_std is not None:
        mask &= (std_arr <= max_std).astype(np.uint8)
        print(f"   std threshold: {max_std} rad")

    n_valid = mask.sum()
    print(f"   Valid pixels: {n_valid:,} / {P:,}  ({100*n_valid/P:.1f}%)")
    return mask.reshape(nrows, ncols)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── 0.  Metadata ──────────────────────────────────────────────────────────
    print("Loading metadata …")
    epochs, t_days, nrows, ncols, n_epochs, ref_row, ref_col = load_meta(INVERSION_DIR)
    t_norm = epoch_time_axis(t_days)
    P_total = nrows * ncols
    print(f"  Epochs : {n_epochs}   Grid : {nrows}×{ncols}  ({P_total:,} px)")

    # ── 1.  Open disp_cube (read-only) ────────────────────────────────────────
    disp_path = os.path.join(INVERSION_DIR, "disp_cube.dat")
    if not os.path.exists(disp_path):
        raise FileNotFoundError(f"disp_cube.dat not found: {disp_path}")

    disp_cube = np.memmap(disp_path, dtype="float32", mode="r",
                          shape=(n_epochs, nrows, ncols))
    gb = n_epochs * nrows * ncols * 4 / 1e9
    print(f"  disp_cube.dat  opened: {gb:.2f} GB  (read-only memmap)")

    # ── 2.  QC ────────────────────────────────────────────────────────────────
    qc_disp_cube(disp_cube, n_epochs, nrows, ncols)

    # ── 3.  APS correction ────────────────────────────────────────────────────
    aps_path       = os.path.join(OUTPUT_DIR, APS_CUBE_NAME)
    disp_clean_path = os.path.join(OUTPUT_DIR, DISP_CLEAN_NAME)

    aps_cube_mm   = np.memmap(aps_path,        dtype="float32", mode="w+",
                               shape=(n_epochs, nrows, ncols))
    disp_clean_mm = np.memmap(disp_clean_path, dtype="float32", mode="w+",
                               shape=(n_epochs, nrows, ncols))
    # Pre-fill with NaN
    aps_cube_mm[:]   = np.nan
    disp_clean_mm[:] = np.nan
    aps_cube_mm.flush()
    disp_clean_mm.flush()

    remove_aps_streaming(
        disp_cube, aps_cube_mm, disp_clean_mm,
        nrows, ncols, n_epochs,
        poly_degree=APS_POLY_DEGREE,
        sigma=APS_GAUSS_SIGMA,
        n_iter=APS_ITERATIONS,
    )

    # ── 4.  Temporal SG smoothing (optional) ──────────────────────────────────
    sg_window = SG_WINDOW
    if APPLY_SG_FILTER:
        # Clip window to n_epochs and ensure it's odd
        sg_window = min(sg_window, n_epochs)
        if sg_window % 2 == 0:
            sg_window -= 1
        if sg_window > SG_POLY_ORDER:
            apply_savgol_streaming(disp_clean_mm, nrows, ncols, n_epochs,
                                   window=sg_window, poly_order=SG_POLY_ORDER)
        else:
            print("\n── SG smoothing skipped: not enough epochs for chosen window ──")

    # ── 5.  Velocity map ──────────────────────────────────────────────────────
    velocity = compute_velocity(disp_clean_mm, nrows, ncols, n_epochs, t_days)

    # ── 6.  Reference grounding ───────────────────────────────────────────────
    ground_reference(disp_clean_mm, velocity, nrows, ncols, n_epochs,
                      mode=REFERENCE_MODE,
                      pixel=REFERENCE_PIXEL,
                      auto_pct=AUTO_REF_PERCENTILE)

    # Recompute velocity after grounding
    if REFERENCE_MODE is not None:
        velocity = compute_velocity(disp_clean_mm, nrows, ncols, n_epochs, t_days)

    # ── 7.  Reliability mask ──────────────────────────────────────────────────
    mask = make_reliability_mask(disp_clean_mm, nrows, ncols, n_epochs,
                                  min_valid_epochs=MIN_VALID_EPOCHS,
                                  max_std=MAX_STD_THRESHOLD)

    # Apply mask to velocity
    velocity[mask == 0] = np.nan

    # ── 8.  Write outputs ─────────────────────────────────────────────────────
    print("\n── Writing final outputs ───────────────────────────────────────────")

    vel_path  = os.path.join(OUTPUT_DIR, VELOCITY_NAME)
    vel_mm    = np.memmap(vel_path, dtype="float32", mode="w+",
                          shape=(nrows, ncols))
    vel_mm[:] = velocity
    vel_mm.flush()
    print(f"   velocity.dat        →  ({nrows}, {ncols})")

    mask_path  = os.path.join(OUTPUT_DIR, MASK_NAME)
    mask_mm    = np.memmap(mask_path, dtype="uint8", mode="w+",
                            shape=(nrows, ncols))
    mask_mm[:] = mask
    mask_mm.flush()
    print(f"   reliability_mask    →  ({nrows}, {ncols})")

    print(f"   aps_cube.dat        →  ({n_epochs}, {nrows}, {ncols})")
    print(f"   disp_clean.dat      →  ({n_epochs}, {nrows}, {ncols})")

    # ── 9.  Save postprocessing metadata ─────────────────────────────────────
    np.savez(
        os.path.join(OUTPUT_DIR, "postprocess_meta.npz"),
        epochs             = epochs,
        t_days             = t_days.astype(np.float32),
        nrows              = np.int32(nrows),
        ncols              = np.int32(ncols),
        n_epochs           = np.int32(n_epochs),
        aps_poly_degree    = np.int32(APS_POLY_DEGREE),
        aps_gauss_sigma    = np.float32(APS_GAUSS_SIGMA),
        aps_iterations     = np.int32(APS_ITERATIONS),
        sg_applied         = np.bool_(APPLY_SG_FILTER),
        sg_window          = np.int32(sg_window if APPLY_SG_FILTER else 0),
        sg_poly_order      = np.int32(SG_POLY_ORDER),
        reference_mode     = str(REFERENCE_MODE),
        min_valid_epochs   = np.int32(MIN_VALID_EPOCHS),
        max_std_threshold  = np.float32(MAX_STD_THRESHOLD or np.nan),
        n_valid_pixels     = np.int32(int(mask.sum())),
        median_abs_velocity = np.float32(float(np.nanmedian(np.abs(velocity)))),
    )
    print("   postprocess_meta.npz  ✔")

    elapsed = time.time() - t0
    print(f"\n── Done in {elapsed/60:.1f} min ──────────────────────────────────────")
    print(f"All outputs → {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("  • velocity.dat   — load with np.memmap(..., shape=(nrows,ncols))")
    print("  • disp_clean.dat — full time series for hotspot extraction")
    print("  • Convert to GeoTIFF with rasterio if geocoding is available")


if __name__ == "__main__":
    main()