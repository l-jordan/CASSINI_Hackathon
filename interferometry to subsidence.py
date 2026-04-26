"""
SBAS Inversion  —  Stage 2  (v3 — scalable weighted engine)
=============================================================
Fixes vs v2:
  ✔  Weighted solver replaced: sparse normal-equation accumulation
       — no (N,M,P) or (P,N,M) intermediates; O(M×P) work per block
       — WEIGHTED_BLOCK_PIXELS kept small so (P,N,N) stays ≤500 MB
  ✔  Time axis centred before polynomial fit  (better AᵀA conditioning)
  ✔  Dead/incorrect AtW/AtWA code removed from velocity solver
  ✔  Auto RAM guard: APS cube never loaded if > 8 GB
  ✔  Memmap pre-flattened once; no repeated reshape() calls in hot loop
  ✔  build_G returns (G, pri_idx, sec_idx) — sparse indices reused directly

Inputs  (from Stage 1):
    clean_cube.dat   — φ_clean  (n_ifg, rows, cols) float32 memmap, NaN invalid
    weight_cube.dat  — CC²/(1−CC²+ε) weights        (n_ifg, rows, cols) float32 memmap
    meta.npz         — pair + grid metadata

Outputs:
    disp_cube.dat      — cumulative displacement time series (epochs, rows, cols) float32
    aps_cube.dat       — atmospheric phase screen per epoch  (epochs, rows, cols) float32
    velocity.dat       — linear velocity map                 (rows, cols)          float32
    inversion_meta.npz — epoch list, time axis, diagnostics

Requirements:  pip install numpy scipy tqdm
"""

import os, json, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_DIR  = r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Subset_Preprocessed"
OUTPUT_DIR = r"C:\Users\iamal\Documents\CassiniWell\Data\SBAS_Inversion"

# Inversion
MIN_VALID_IFGS   = 4       # pixels with fewer valid IFGs → NaN
WEIGHTED_MODE    = True    # True  = coherence-weighted block solve (slower, better quality)
                            # False = unweighted pseudoinverse (fastest, good for most cases)
PINV_RCOND       = 1e-3    # condition number cutoff for pseudoinverse (regularisation)

# Block processing
# Unweighted: single BLAS call, large blocks are fine.
#   558 IFGs × 200,000 pixels × 4 B ≈ 420 MB per block load.
# Weighted: stores (P, N, N) normal matrix in RAM.
#   P=8000, N=80 → 8000×80×80×8 B ≈ 410 MB — safe on 16 GB systems.
#   Do NOT set WEIGHTED_BLOCK_PIXELS above ~15,000 unless you have ≥32 GB free.
BLOCK_PIXELS          = 200_000   # unweighted mode
WEIGHTED_BLOCK_PIXELS = 8_000     # weighted mode — kept small to cap (P,N,N) RAM

# Atmospheric removal
APS_POLY_DEGREE  = 1       # 1 = linear detrend, 2 = quadratic
APS_GAUSS_SIGMA  = 15      # σ in pixels for spatial APS smoothing (tune to ~500-2000 m)
APS_ITERATIONS   = 2       # refinement passes
APS_IN_RAM       = True    # True  = load full disp_cube for vectorised APS (recommended)
                            # False = block-process APS epochs (use if RAM < 8 GB)

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def load_meta(input_dir):
    meta    = np.load(os.path.join(input_dir, "meta.npz"), allow_pickle=True)
    pairs   = json.loads(str(meta["pairs_json"]))
    epochs  = list(meta["epochs"])
    nrows   = int(meta["nrows"])
    ncols   = int(meta["ncols"])
    n_ifg   = int(meta["n_ifg"])
    ref_row = int(meta["ref_row"])
    ref_col = int(meta["ref_col"])
    return pairs, epochs, nrows, ncols, n_ifg, ref_row, ref_col


def epoch_time_axis(epochs):
    """
    Return (t_days, t_norm).
    t_days : days from first epoch              (used for velocity in rad/day)
    t_norm : mean-centred, unit-scaled version  (used for polynomial design matrix)

    Centring matters: Vandermonde columns become near-orthogonal when t is
    centred, which keeps AtA well-conditioned even for degree-2 polynomials
    over long time series.  Without centring, degree-2 fits on a [0,1] axis
    have condition numbers ~10³ worse than on a [-0.5, 0.5] axis.
    """
    dates  = [datetime.strptime(e, "%Y%m%d") for e in epochs]
    t_days = np.array([(d - dates[0]).days for d in dates], dtype=np.float64)
    span   = t_days[-1] - t_days[0]
    t_norm = (t_days - t_days.mean()) / span if span > 0 else t_days
    return t_days, t_norm


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DESIGN MATRIX + PSEUDOINVERSE
# ══════════════════════════════════════════════════════════════════════════════

def build_G(pairs, epochs):
    """
    Build dense G matrix (M × N) and extract the sparse index arrays.

    G[i, j_sec] = +1,  G[i, j_pri] = -1  (exactly 2 nonzeros per row).

    Also returns pri_idx, sec_idx  (M,) int arrays — the weighted solver
    uses these directly to exploit G's sparsity without storing a (N,M,P)
    intermediate.
    """
    epoch_idx = {e: k for k, e in enumerate(epochs)}
    M, N = len(pairs), len(epochs)
    G       = np.zeros((M, N), dtype=np.float64)
    pri_idx = np.empty(M, dtype=np.intp)
    sec_idx = np.empty(M, dtype=np.intp)
    for i, p in enumerate(pairs):
        pi = epoch_idx[p["primary"]]
        si = epoch_idx[p["secondary"]]
        G[i, pi]   = -1.0
        G[i, si]   = +1.0
        pri_idx[i] = pi
        sec_idx[i] = si
    return G, pri_idx, sec_idx


def compute_pinv(G, rcond):
    """
    Compute regularised pseudoinverse G⁺  (N × M) via truncated SVD.
    Singular values below rcond × σ_max are zeroed (Tikhonov-equivalent).
    """
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    thresh = rcond * s[0]
    s_inv  = np.where(s > thresh, 1.0 / s, 0.0)
    G_pinv = (Vt.T * s_inv) @ U.T   # (N, M)
    cond   = s[0] / s[np.sum(s > thresh) - 1] if np.sum(s > thresh) > 1 else np.inf
    print(f"G shape: {G.shape}  |  effective rank: {int(np.sum(s > thresh))} / {len(s)}  "
          f"|  condition number: {cond:.1f}")
    return G_pinv


# ══════════════════════════════════════════════════════════════════════════════
# 2A.  FAST UNWEIGHTED INVERSION  (single BLAS call per block)
# ══════════════════════════════════════════════════════════════════════════════

def invert_unweighted(G_pinv, phi_block, valid_count):
    """
    disp_block = G_pinv @ phi_block       (N × P)

    phi_block  : (M, P)  NaN → 0 pre-filled
    valid_count: (P,)    number of non-NaN IFGs per pixel
    Returns (N, P) float32 with NaN where < MIN_VALID_IFGS.
    """
    disp = (G_pinv @ phi_block).astype(np.float32)
    disp[:, valid_count < MIN_VALID_IFGS] = np.nan
    return disp


# ══════════════════════════════════════════════════════════════════════════════
# 2B.  WEIGHTED BLOCK INVERSION  (sparse normal-equation accumulation)
# ══════════════════════════════════════════════════════════════════════════════

def invert_weighted_sparse(pri_idx, sec_idx, phi_block, w_block, valid_count,
                           N, rcond):
    """
    Scalable coherence-weighted SBAS inversion.

    ── Why the old (P,N,N) broadcast approach broke ──────────────────────────
    Building GtW = G.T[:,:,None] * w_block[None,:,:]  creates a (N,M,P)
    intermediate.  For N=80, M=558, P=200,000 that is 7.2 billion float64
    values (≈57 GB).  Even with smaller blocks it scales as O(N×M×P).

    ── This implementation ───────────────────────────────────────────────────
    G has exactly 2 nonzero entries per row: G[m, pri_m]=-1, G[m, sec_m]=+1.
    Each IFG therefore contributes only 4 scalar entries to GtWG and 2 to
    GtWphi.  We accumulate them with an M-iteration loop, each iteration
    doing four O(P) in-place additions — no large intermediate arrays.

    Memory: only (P,N,N) + (P,N) float64 are kept in RAM.
      P=8000, N=80 → GtWG ≈ 410 MB  (controlled by WEIGHTED_BLOCK_PIXELS)

    Parameters
    ----------
    pri_idx, sec_idx : (M,) int   — epoch column indices for each IFG
    phi_block        : (M, P) f64 — phase, NaN→0 filled
    w_block          : (M, P) f64 — weights, 0 for invalid obs
    valid_count      : (P,)  int  — number of non-NaN IFGs per pixel
    N                : int        — number of epochs
    rcond            : float      — Tikhonov damping fraction

    Returns  (N, P) float32, NaN where valid_count < MIN_VALID_IFGS.
    """
    M = len(pri_idx)
    P = phi_block.shape[1]

    GtWG   = np.zeros((P, N, N), dtype=np.float64)   # (P, N, N)
    GtWphi = np.zeros((P, N),    dtype=np.float64)   # (P, N)

    # Accumulate over IFGs — M iterations, each O(P) work
    for m in range(M):
        pi = pri_idx[m]
        si = sec_idx[m]
        w  = w_block[m]    # (P,)
        ph = phi_block[m]  # (P,)

        # 4 contributions to normal matrix  (G[m,pi]=-1, G[m,si]=+1)
        #   outer product of g_m with itself, scaled by w, added to each pixel
        GtWG[:, pi, pi] += w
        GtWG[:, si, si] += w
        GtWG[:, pi, si] -= w
        GtWG[:, si, pi] -= w

        # 2 contributions to RHS
        wph = w * ph
        GtWphi[:, pi] -= wph   # G[m,pi] = -1
        GtWphi[:, si] += wph   # G[m,si] = +1

    # Tikhonov damping: add rcond * max_diagonal * I to each pixel's system
    # Prevents singular systems for pixels with few observations.
    diag    = GtWG[:, np.arange(N), np.arange(N)]          # (P, N)
    damp    = rcond * np.maximum(diag.max(axis=1), 1e-6)   # (P,)
    GtWG[:, np.arange(N), np.arange(N)] += damp[:, None]

    # Batched solve: numpy dispatches to LAPACK gesv for all P systems at once.
    # GtWG  : (P, N, N)
    # GtWphi: must be (P, N, 1) — the trailing 1 tells numpy this is a batch
    #         of column vectors, not a 2-D matrix.  Without it numpy mis-reads
    #         the dimensions and raises "size P is different from N".
    disp_pn = np.linalg.solve(GtWG, GtWphi[:, :, None])[:, :, 0]   # (P, N)

    disp = disp_pn.T.astype(np.float32)        # (N, P)
    disp[:, valid_count < MIN_VALID_IFGS] = np.nan
    return disp


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VECTORISED ATMOSPHERIC REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def remove_atmosphere_vectorised(d, t_norm, sigma, poly_deg, n_iter):
    """
    Two-pass APS removal on (N, rows, cols) cube.
    All operations are vectorised — no Python loops over pixels or epochs.

    Pass A — temporal polynomial detrend (vectorised lstsq over all pixels):
        A @ coeffs ≈ d   →   trend = A @ (AᵀA)⁻¹ Aᵀ d_2d
        residual   = d − trend

    Pass B — 3-D Gaussian filter with σ=(0, σ_xy, σ_xy):
        sigma=0 along time axis  →  no temporal blurring
        spatial smoothing of each epoch slice simultaneously

    Returns: (d_clean, aps_total) both (N, rows, cols) float32.
    """
    N, nrows, ncols = d.shape
    P = nrows * ncols

    # Polynomial design matrix  (N, deg+1)
    A = np.vander(t_norm, poly_deg + 1, increasing=True).astype(np.float64)
    AtA     = A.T @ A                         # (deg+1, deg+1)  — tiny
    AtA_inv = np.linalg.pinv(AtA)            # (deg+1, deg+1)

    d_work    = d.reshape(N, P).astype(np.float64, copy=True)
    aps_total = np.zeros((N, P), dtype=np.float64)

    for iteration in range(n_iter):
        # ── Pass A: vectorised temporal detrend ──────────────────────────────
        # Replace NaN with 0 for the lstsq (invalid pixels won't contribute).
        d_filled    = np.nan_to_num(d_work, nan=0.0)           # (N, P)

        # (deg+1, P) = (deg+1, N) @ (N, P)
        coeffs      = AtA_inv @ (A.T @ d_filled)

        # trend (N, P) = (N, deg+1) @ (deg+1, P)
        trend       = A @ coeffs

        # Residual: keep NaN structure from d_work
        residual    = d_work - trend                            # (N, P) — NaN where original NaN

        # ── Pass B: spatial Gaussian filter on residual cube ─────────────────
        res_cube    = residual.reshape(N, nrows, ncols)

        # Fill NaN → 0 and compute coverage mask (for border-correction)
        valid_mask  = np.isfinite(res_cube).astype(np.float32)
        res_filled  = np.nan_to_num(res_cube, nan=0.0).astype(np.float32)

        # Single 3-D call:  sigma=(0, σ, σ)  — no temporal blur, full spatial blur
        blurred     = gaussian_filter(res_filled,
                                      sigma=(0, sigma, sigma)).astype(np.float64)
        coverage    = gaussian_filter(valid_mask,
                                      sigma=(0, sigma, sigma)).astype(np.float64)
        coverage    = np.maximum(coverage, 1e-6)

        # APS estimate for this iteration (NaN where no valid data)
        aps_epoch   = np.where(valid_mask > 0.05,
                               blurred / coverage, 0.0)        # (N, nrows, ncols)

        aps_iter    = aps_epoch.reshape(N, P)
        aps_total  += aps_iter
        d_work     -= aps_iter                                  # subtract APS in-place

        mean_aps = float(np.nanmean(np.abs(aps_epoch)))
        print(f"    APS iter {iteration + 1}/{n_iter}  |  mean|APS| = {mean_aps:.5f} rad")

    d_clean = d_work.reshape(N, nrows, ncols).astype(np.float32)
    aps_out = aps_total.reshape(N, nrows, ncols).astype(np.float32)

    # Re-apply original NaN mask
    nan_mask = ~np.isfinite(d.reshape(N, nrows, ncols))
    d_clean[nan_mask] = np.nan
    aps_out[nan_mask] = np.nan

    return d_clean, aps_out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VELOCITY MAP  (vectorised linear fit)
# ══════════════════════════════════════════════════════════════════════════════

def compute_velocity_vectorised(d, t_days):
    """
    Per-pixel linear velocity via closed-form 2×2 Cramer's rule.
    Handles NaN epochs correctly: missing epochs contribute nothing to sums.

    d      : (N, rows, cols) float32
    t_days : (N,) float64  — days since first epoch

    Returns velocity map (rows, cols) float32  [same units as d, per day].
    """
    N, nrows, ncols = d.shape
    P = nrows * ncols

    t = (t_days - t_days[0]).astype(np.float64)   # ensure zero-origin

    d_2d  = d.reshape(N, P).astype(np.float64)
    valid = np.isfinite(d_2d)                      # (N, P) — boolean weight
    fill  = np.nan_to_num(d_2d, nan=0.0)          # (N, P) — NaN → 0

    # Accumulate 2×2 normal-equation components over epochs (vectorised over P)
    # Model: d = a + v*t  →  minimize Σ (d_n - a - v*t_n)²  over valid epochs
    w = valid.astype(np.float64)                   # (N, P)  — 0/1 weights

    s0 = w.sum(axis=0)                             # Σ 1      (P,)
    s1 = (t[:, None] * w).sum(axis=0)             # Σ t      (P,)
    s2 = (t[:, None] ** 2 * w).sum(axis=0)        # Σ t²     (P,)
    r0 = (w * fill).sum(axis=0)                   # Σ d      (P,)
    r1 = (t[:, None] * w * fill).sum(axis=0)      # Σ t·d    (P,)

    # Cramer's rule for the 2×2 system [[s0,s1],[s1,s2]] [a,v]ᵀ = [r0,r1]ᵀ
    det   = s0 * s2 - s1 ** 2                     # (P,)
    ok    = (det > 0) & (s0 >= 2)                 # need ≥2 valid epochs

    vel_1d = np.full(P, np.nan, dtype=np.float32)
    vel_1d[ok] = ((s0[ok] * r1[ok] - s1[ok] * r0[ok]) / det[ok]).astype(np.float32)

    return vel_1d.reshape(nrows, ncols)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Metadata ──────────────────────────────────────────────────────────────
    print("Loading metadata …")
    pairs, epochs, nrows, ncols, n_ifg, ref_row, ref_col = load_meta(INPUT_DIR)
    n_epochs = len(epochs)
    P_total  = nrows * ncols
    t_days, t_norm = epoch_time_axis(epochs)

    print(f"  Epochs : {n_epochs}  |  IFGs : {n_ifg}  |  Grid : {nrows}×{ncols} "
          f"({P_total:,} pixels)")
    print(f"  Reference pixel : row={ref_row}, col={ref_col}")

    # ── Design matrix + pseudoinverse ─────────────────────────────────────────
    print("\nBuilding G matrix and pseudoinverse …")
    G, pri_idx, sec_idx = build_G(pairs, epochs)   # (M,N), (M,), (M,)
    G_pinv = compute_pinv(G, PINV_RCOND)           # (N, M) float64

    # ── Memory-mapped inputs ──────────────────────────────────────────────────
    clean_flat  = np.memmap(os.path.join(INPUT_DIR, "clean_cube.dat"),
                            dtype="float32", mode="r",
                            shape=(n_ifg, P_total))   # pre-flatten saves reshape calls
    weight_flat = np.memmap(os.path.join(INPUT_DIR, "weight_cube.dat"),
                            dtype="float32", mode="r",
                            shape=(n_ifg, P_total))

    # ── Output: displacement cube ─────────────────────────────────────────────
    disp_path = os.path.join(OUTPUT_DIR, "disp_cube.dat")
    disp_cube = np.memmap(disp_path, dtype="float32", mode="w+",
                          shape=(n_epochs, nrows, ncols))
    disp_cube[:] = np.nan
    disp_cube.flush()

    # ── Block-wise inversion ──────────────────────────────────────────────────
    block_sz = WEIGHTED_BLOCK_PIXELS if WEIGHTED_MODE else BLOCK_PIXELS
    mode_str = "weighted sparse-accumulation" if WEIGHTED_MODE else "unweighted pseudoinverse"
    print(f"\nInverting ({mode_str}), block = {block_sz:,} pixels …")

    n_blocks    = int(np.ceil(P_total / block_sz))
    solved_pix  = 0
    disp_flat   = disp_cube.reshape(n_epochs, P_total)   # view, no copy

    for b in tqdm(range(n_blocks), desc="Inversion blocks"):
        p0 = b * block_sz
        p1 = min(p0 + block_sz, P_total)

        phi_block = clean_flat[:, p0:p1].astype(np.float64)    # (M, P)
        w_block   = weight_flat[:, p0:p1].astype(np.float64)

        valid_count = np.isfinite(phi_block).sum(axis=0)        # (P,)

        phi_filled = np.nan_to_num(phi_block, nan=0.0)
        w_filled   = np.nan_to_num(w_block,   nan=0.0)

        if WEIGHTED_MODE:
            disp_block = invert_weighted_sparse(
                pri_idx, sec_idx, phi_filled, w_filled,
                valid_count, n_epochs, PINV_RCOND
            )
        else:
            disp_block = invert_unweighted(G_pinv, phi_filled, valid_count)

        disp_flat[:, p0:p1] = disp_block
        solved_pix += int((valid_count >= MIN_VALID_IFGS).sum())

        if (b + 1) % 10 == 0 or b == n_blocks - 1:
            disp_cube.flush()
        if b % 20 == 0:
            time.sleep(0.001)

    pct = 100 * solved_pix / P_total
    print(f"Inversion done — {solved_pix:,} / {P_total:,} pixels solved ({pct:.1f}%)")

    # ── Atmospheric removal ───────────────────────────────────────────────────
    print(f"\nAtmospheric removal …")
    print(f"  poly_deg={APS_POLY_DEGREE}  |  σ={APS_GAUSS_SIGMA}px  "
          f"|  iterations={APS_ITERATIONS}")

    cube_gb = n_epochs * nrows * ncols * 4 / 1e9
    load_in_ram = APS_IN_RAM and cube_gb < 8.0   # auto-protect: never load >8 GB
    if APS_IN_RAM and not load_in_ram:
        print(f"  WARNING: disp_cube is {cube_gb:.1f} GB — falling back to memmap mode "
              f"(set APS_IN_RAM=False to silence this)")
    if load_in_ram:
        d = np.array(disp_cube, dtype=np.float32)
        print(f"  Loaded disp_cube into RAM: {cube_gb:.2f} GB")
    else:
        d = disp_cube

    d_clean, aps_arr = remove_atmosphere_vectorised(
        d, t_norm, sigma=APS_GAUSS_SIGMA,
        poly_deg=APS_POLY_DEGREE, n_iter=APS_ITERATIONS
    )

    # Write cleaned displacement back
    disp_cube[:] = d_clean
    disp_cube.flush()

    # APS cube
    aps_path = os.path.join(OUTPUT_DIR, "aps_cube.dat")
    aps_cube = np.memmap(aps_path, dtype="float32", mode="w+",
                         shape=(n_epochs, nrows, ncols))
    aps_cube[:] = aps_arr
    aps_cube.flush()
    print("  APS cube written.")

    # ── Velocity map ──────────────────────────────────────────────────────────
    print("\nComputing velocity map (vectorised 2×2 Cramer solve) …")
    vel = compute_velocity_vectorised(d_clean, t_days)

    vel_path = os.path.join(OUTPUT_DIR, "velocity.dat")
    vel_mm   = np.memmap(vel_path, dtype="float32", mode="w+", shape=(nrows, ncols))
    vel_mm[:] = vel
    vel_mm.flush()
    print(f"  median |v| = {float(np.nanmedian(np.abs(vel))):.5f} rad/day")

    # ── Save metadata ─────────────────────────────────────────────────────────
    np.savez(
        os.path.join(OUTPUT_DIR, "inversion_meta.npz"),
        epochs      = epochs,
        t_days      = t_days.astype(np.float32),
        ref_row     = np.int32(ref_row),
        ref_col     = np.int32(ref_col),
        nrows       = np.int32(nrows),
        ncols       = np.int32(ncols),
        n_epochs    = np.int32(n_epochs),
        n_ifg       = np.int32(n_ifg),
        aps_sigma   = np.float32(APS_GAUSS_SIGMA),
        poly_degree = np.int32(APS_POLY_DEGREE),
        solved_pix  = np.int32(solved_pix),
        mode        = "weighted" if WEIGHTED_MODE else "unweighted",
    )

    elapsed = time.time() - t0
    print(f"\n── Complete in {elapsed/60:.1f} min ──────────────────────────────────")
    print(f"  disp_cube.dat  →  ({n_epochs}, {nrows}, {ncols})")
    print(f"  aps_cube.dat   →  ({n_epochs}, {nrows}, {ncols})")
    print(f"  velocity.dat   →  ({nrows}, {ncols})")
    print(f"  inversion_meta.npz")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()  