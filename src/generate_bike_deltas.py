"""Generate bike+rider injection deltas for all UT-LUMPI frames (run5 input).

Per frame: synthesized composite points (Nx3 float32 .npy) + Bicycle/
Motorcycle label rows (.txt), stored under bike_deltas/. Merged with scene
points and surgery labels at training-prep time.

Run: .venv/bin/python src/generate_bike_deltas.py [start] [end]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import bike_synth
from pedestrian_synth import SensorModel

PCD = ROOT / "data/ut_lumpi_synthetic/UCF-DT-LUMPI-Seq-1"
OUT = PCD / "bike_deltas"

_sm = _prior = None


def _init():
    global _sm, _prior
    _sm = SensorModel()
    d = np.load(ROOT / "tracks/lumpi/bike_stats.npz")
    b = d["bicycle"]
    _prior = {"pos": b[:, :2], "dims": b[:, 2:5]}


def one(name: str):
    out_pts = OUT / f"{name}.npy"
    out_lbl = OUT / f"{name}.txt"
    if out_pts.exists() and out_lbl.exists():
        return 0
    rng = np.random.default_rng(950_000 + int(name))
    scene = np.load(PCD / "pcd" / f"{name}.npy").astype(np.float64)
    labels = (PCD / "lbl" / f"{name}.txt").read_text().splitlines()
    pts_list, rows = bike_synth.synthesize_bikes(scene, labels, _sm.sensors, _prior, None, rng)
    pts = (np.concatenate(pts_list) if pts_list else np.zeros((0, 3), dtype=np.float32))
    np.save(out_pts, pts.astype(np.float32))
    out_lbl.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def main():
    OUT.mkdir(exist_ok=True)
    names = sorted(p.stem for p in (PCD / "pcd").glob("*.npy"))
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(names)
    names = names[lo:hi]
    with Pool(8, initializer=_init) as pool:
        counts = []
        for i, c in enumerate(pool.imap_unordered(one, names, chunksize=20)):
            counts.append(c)
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(names)}", flush=True)
    print(f"done: {len(counts)} frames, {sum(counts)} bikes injected")


if __name__ == "__main__":
    main()
