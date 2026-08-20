"""run3 label surgery v3: matched-pair corrected factors for UT-LUMPI labels.

Replaces the naive real/synthetic surgery. Lesson from the v5 server result
(2026-07-22): the naive real-median/syn-mean factors DOUBLE-COUNT the
model's intrinsic ~1.13x up-correction over its training labels, overshooting
the hidden slice (run2 Car: predicted 5.15m vs run1's 4.08m; server Car
0.568 -> 0.297). The principled factor per class is measured as
GT_median / run1_predicted_median on matched pairs (BEV IoU>0.25) on the
harness — the multiplier that makes run1's own predictions land on real
sizes, no double-counting.

Per-class factors applied to ORIGINAL synthetic labels:
  Car (1.088,1.264,1.183)  — robust, n=2440 pairs (naive was 1.23/1.31/1.34)
  Bicycle (1.618,2.004,1.477) — n=53 (naive was ~same)
  Motorcycle (1.26,1.31,1.47) — n=1, keep naive estimate
  Truck (1.53,1.35,1.37) — corrected-stats to real 8.45x2.89x3.33 (matched
    pairs too selection-biased to trust: n=42, ratio skewed by unmatched
    small trucks)
  Bus (1.39,0.89,0.92) — corrected-stats to real 11.17x2.74x3.07
  Van (1.08,1.24,1.08) — mild, to ~4.69x1.99x1.98
  Person: NO factor — labels stay at synthesized real stats; its server gap
    is score/threshold-driven (7.6x overprediction), not size-driven.

Method unchanged: rescale, bottom-anchored, synthetic-only (legal DA).
Output: data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/lbl_surgery3/
Run: .venv/bin/python src/label_surgery3.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/lbl"
DST = ROOT / "data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1/lbl_surgery3"

SCALE = {
    "Car":        (1.088, 1.264, 1.183),
    "Bicycle":    (1.618, 2.004, 1.477),
    "Motorcycle": (1.26, 1.31, 1.47),
    "Truck":      (1.53, 1.35, 1.37),
    "Bus":        (1.39, 0.89, 0.92),
    "Van":        (1.08, 1.24, 1.08),
}


def main() -> None:
    DST.mkdir(exist_ok=True)
    n_boxes = n_scaled = 0
    for lp in sorted(SRC.glob("*.txt")):
        out = []
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 8:
                continue
            n_boxes += 1
            cls = f[7]
            if cls in SCALE:
                sx, sy, sz = SCALE[cls]
                x, y, z, dx, dy, dz, hd = (float(v) for v in f[:7])
                ndx, ndy, ndz = dx * sx, dy * sy, dz * sz
                z = z - dz / 2 + ndz / 2  # keep bottom fixed
                f[:7] = [f"{v:.4f}" for v in (x, y, z, ndx, ndy, ndz, hd)]
                n_scaled += 1
            out.append(" ".join(f))
        (DST / lp.name).write_text("\n".join(out) + "\n")
    print(f"rewrote {n_boxes} boxes ({n_scaled} scaled) into {DST}")
    for c, s in SCALE.items():
        print(f"  {c:11s} scale l,w,h = {s}")


if __name__ == "__main__":
    main()
