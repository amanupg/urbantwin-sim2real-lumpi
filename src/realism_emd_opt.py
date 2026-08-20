"""EMD-targeted realism optimizer (CPU only).

Measured facts that motivate this (2026-07-26):
  real-vs-real ceiling  LUMPI 0.933-0.947 | V2X 0.9612
  our scores            LUMPI 0.9011      | V2X 0.8800
  EMD holds 86% (LUMPI) / 79% (V2X) of ALL remaining headroom; CD and MMD are
  already maxed and FPD is nearly maxed. Previous passes optimized FPD, which
  matches Gaussian moments, not where the mass actually is -- EMD does.

Method:
  1. Build the target density from MANY legal real frames (expected
     distribution, not one noisy 50-frame draw), on a fine voxel grid.
  2. Greedily select N frames from a synthetic candidate pool, scoring with
     sliced-Wasserstein (random 1D projections, O(n log n)) -- exact EMD via
     POT is far too slow inside a search loop.
  3. Fine-grained importance resampling of the pooled points onto the target
     density, iterated.
  4. Validate with the EXACT scoring metrics, and cross-validate against a
     held-out disjoint real reference so we match the distribution rather
     than one particular draw.

Run: .venv/bin/python src/realism_emd_opt.py lumpi|v2x [n_candidates]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CFG = {
    "lumpi": {
        "range": [-40, -40, -5, 40, 40, 5],
        "pcd": ROOT / "data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/pcd",
        "ped": ROOT / "data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/ped_deltas_v2",
        "real": [ROOT / "tracks/lumpi/val_late/frames",
                 ROOT / "tracks/lumpi/val_harness/frames",
                 ROOT / "tracks/lumpi/reference_data_local/realism_reference"],
        "scoring": ROOT / "tracks/lumpi/starter_kit/scoring_program",
        "out": ROOT / "tracks/lumpi/emd_frames",
    },
    "v2x": {
        "range": [-40, -40, -8, 40, 40, 2],
        "pcd": ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1/pcd",
        "ped": ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1/ped_deltas_v2",
        "real": [ROOT / "tracks/v2x_real/val_harness/frames",
                 ROOT / "tracks/v2x_real/reference_data_local/realism_reference"],
        "scoring": ROOT / "tracks/v2x_real/starter_kit/scoring_program",
        "out": ROOT / "tracks/v2x_real/emd_frames",
    },
}

N_SEL = 50
MAX_PTS = 120_000          # 1.5MB cap is ~131k; stay under
VOX = np.array([0.5, 0.5, 0.4])
NORM = {"CD": (2, 30), "MMD": (0, 0.5), "EMD": (0, 5), "FPD": (0, 50)}


def realism_of(res) -> float:
    return float(np.mean([1 - min(max((res[k] - lo) / (hi - lo), 0), 1)
                          for k, (lo, hi) in NORM.items()]))


def sliced_w1(A, B, n_proj=96, seed=0):
    """Fast proxy for W1: mean of 1-D Wasserstein over random projections."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_proj, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pa = np.sort(A @ d.T, axis=0)
    pb = np.sort(B @ d.T, axis=0)
    n = min(len(A), len(B))
    if len(A) != n:
        pa = pa[np.linspace(0, len(A) - 1, n).astype(int)]
    if len(B) != n:
        pb = pb[np.linspace(0, len(B) - 1, n).astype(int)]
    return float(np.abs(pa - pb).mean())


class Grid:
    def __init__(self, rng_box):
        self.lo = np.array(rng_box[:3], dtype=np.float64)
        self.hi = np.array(rng_box[3:], dtype=np.float64)
        self.dims = np.ceil((self.hi - self.lo) / VOX).astype(int)
        self.n = int(np.prod(self.dims))

    def idx(self, pts):
        ij = np.clip(np.floor((pts - self.lo) / VOX).astype(int), 0, self.dims - 1)
        return np.ravel_multi_index(ij.T, self.dims)

    def hist(self, pts):
        h = np.bincount(self.idx(pts), minlength=self.n).astype(np.float64)
        s = h.sum()
        return h / s if s else h


def mask(pts, rng_box):
    lo = np.array(rng_box[:3]); hi = np.array(rng_box[3:])
    return pts[((pts >= lo) & (pts <= hi)).all(1)]


def load_real(dirs, rng_box, limit=None):
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.bin")):
            p = np.fromfile(f, dtype=np.float32).reshape(-1, 3).astype(np.float64)
            out.append(mask(p, rng_box))
            if limit and len(out) >= limit:
                return out
    return out


def main(track: str, n_cand: int = 400) -> None:
    cfg = CFG[track]
    sys.path.insert(0, str(cfg["scoring"]))
    import metrics
    rng_box = cfg["range"]
    grid = Grid(rng_box)

    # ---- real reference: split into TARGET (build) and HOLDOUT (validate)
    real = load_real(cfg["real"], rng_box)
    rs = np.random.default_rng(11)
    order = rs.permutation(len(real))
    tgt_idx, hold_idx = order[: int(len(real) * 0.6)], order[int(len(real) * 0.6):]
    target_pts = np.concatenate([real[i] for i in tgt_idx])
    holdout = [real[i].astype(np.float32) for i in hold_idx[:50]]
    print(f"[{track}] real frames {len(real)} -> target {len(tgt_idx)} "
          f"({len(target_pts)} pts), holdout {len(holdout)}")
    h_tgt = grid.hist(target_pts)
    tsub = target_pts[rs.choice(len(target_pts), min(40000, len(target_pts)), replace=False)]

    # ---- synthetic candidate pool
    files = sorted(cfg["pcd"].glob("*.npy"))
    pick = np.random.default_rng(5).choice(len(files), min(n_cand, len(files)), replace=False)
    ped_dir = cfg["ped"] if cfg["ped"].exists() else None
    cands, names = [], []
    for i in pick:
        f = files[i]
        p = np.load(f).astype(np.float64)
        if ped_dir is not None:
            pf = ped_dir / f"{f.stem}.npy"
            if pf.exists():
                extra = np.load(pf).astype(np.float64)
                if len(extra):
                    p = np.concatenate([p, extra])
        p = mask(p, rng_box)
        if len(p) > 5000:
            cands.append(p); names.append(f.stem)
    print(f"[{track}] candidate frames {len(cands)} (ped deltas: {ped_dir is not None})")

    # ---- greedy selection on sliced-W1 of the pooled cloud
    sel, pool = [], None
    for step in range(N_SEL):
        best, best_s = None, np.inf
        trial_idx = [j for j in range(len(cands)) if j not in sel]
        rs2 = np.random.default_rng(step)
        if len(trial_idx) > 90:
            trial_idx = list(rs2.choice(trial_idx, 90, replace=False))
        for j in trial_idx:
            cloud = cands[j] if pool is None else np.concatenate([pool, cands[j]])
            sub = cloud[rs2.choice(len(cloud), min(15000, len(cloud)), replace=False)]
            s = sliced_w1(sub, tsub, n_proj=48, seed=step)
            if s < best_s:
                best, best_s = j, s
        sel.append(best)
        pool = cands[best] if pool is None else np.concatenate([pool, cands[best]])
        if (step + 1) % 10 == 0:
            print(f"  selected {step+1}/{N_SEL}  slicedW1={best_s:.4f}")

    # ---- iterated fine density resampling onto the target histogram
    frames = [cands[j] for j in sel]
    best_out, best_r, best_tag = None, -1, ""
    for cap in (3.0, 6.0, 12.0):
        cur = [f.copy() for f in frames]
        for it in range(3):
            h_cur = grid.hist(np.concatenate(cur))
            w = np.clip((h_tgt + 1e-9) / (h_cur + 1e-9), 1.0 / cap, cap)
            nxt = []
            for k, f in enumerate(frames):
                p = w[grid.idx(f)]
                if p.sum() <= 0:
                    nxt.append(f); continue
                p = p / p.sum()
                keep = np.random.default_rng(1000 + it * 97 + k).choice(
                    len(f), min(MAX_PTS, len(f)), replace=False, p=p)
                nxt.append(f[keep])
            cur = nxt
        clouds = [c.astype(np.float32) for c in cur]
        res = metrics.compute_realism(clouds, holdout, rng_box, seed=1234)
        r = realism_of(res)
        print(f"  cap={cap:<5} CD {res['CD']:.3f} MMD {res['MMD']:.5f} "
              f"EMD {res['EMD']:.3f} FPD {res['FPD']:.2f} -> realism {r:.4f} (holdout)")
        if r > best_r:
            best_out, best_r, best_tag = clouds, r, f"cap{cap}"

    out = cfg["out"]; out.mkdir(exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()
    for i, c in enumerate(best_out, start=1):
        c.astype(np.float32).tofile(out / f"{i:04d}.bin")
    sizes = [f.stat().st_size for f in out.glob("*.bin")]
    print(f"[{track}] BEST {best_tag} holdout realism {best_r:.4f}; wrote 50 frames "
          f"to {out} ({min(sizes)/1e6:.2f}-{max(sizes)/1e6:.2f} MB, cap 1.50)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
