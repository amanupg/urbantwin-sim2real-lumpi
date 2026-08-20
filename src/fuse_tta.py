"""Fuse TTA views for LUMPI. Runs locally on CPU; the cluster only did stock inference.

Four views: identity, flip-x, flip-y, flip-xy. A flip of the INPUT cloud is undone on the
OUTPUT boxes here:
    flip-x  (x -> -x): cx -> -cx, heading -> pi - heading
    flip-y  (y -> -y): cy -> -cy, heading -> -heading
    flip-xy          : cx,cy -> -cx,-cy, heading -> heading + pi
Boxes are then fused per class with weighted box fusion: cluster by BEV IoU, average
centre/size weighted by score, and set the cluster score to mean(score) * (n_views/4) so
a detection confirmed by all four views outranks one seen only once. That last term is
the entire point of TTA -- it is a CONSENSUS signal, not just an averaging one.

Run: .venv/bin/python src/fuse_tta.py <split>   (split = val | test)
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

VIEWS = ("id", "fx", "fy", "fxy")


def unflip(box, tag):
    b = list(map(float, box))
    if tag == "fx":
        b[0] = -b[0]; b[6] = np.pi - b[6]
    elif tag == "fy":
        b[1] = -b[1]; b[6] = -b[6]
    elif tag == "fxy":
        b[0] = -b[0]; b[1] = -b[1]; b[6] = b[6] + np.pi
    b[6] = (b[6] + np.pi) % (2 * np.pi) - np.pi
    return b


def bev_iou(a, b):
    x1, y1 = max(a[0]-a[3]/2, b[0]-b[3]/2), max(a[1]-a[4]/2, b[1]-b[4]/2)
    x2, y2 = min(a[0]+a[3]/2, b[0]+b[3]/2), min(a[1]+a[4]/2, b[1]+b[4]/2)
    i = max(0, x2-x1) * max(0, y2-y1)
    u = a[3]*a[4] + b[3]*b[4] - i
    return i/u if u > 0 else 0.0


def fuse(dets, iou_thr=0.55, n_views=4):
    dets = sorted(dets, key=lambda d: -d["score"])
    used = [False]*len(dets); out = []
    for i, d in enumerate(dets):
        if used[i]:
            continue
        cluster = [d]; used[i] = True
        for j in range(i+1, len(dets)):
            if used[j] or dets[j]["class"] != d["class"]:
                continue
            if bev_iou(d["box"], dets[j]["box"]) > iou_thr:
                cluster.append(dets[j]); used[j] = True
        w = np.array([c["score"] for c in cluster], dtype=float)
        B = np.array([c["box"][:6] for c in cluster], dtype=float)
        box = list((B * w[:, None]).sum(0) / w.sum())
        # headings are circular -- average as unit vectors, never arithmetically
        h = np.array([c["box"][6] for c in cluster], dtype=float)
        box.append(float(np.arctan2((np.sin(h)*w).sum(), (np.cos(h)*w).sum())))
        nv = len({c.get("_v") for c in cluster})
        out.append({"class": d["class"], "box": box,
                    "score": float(w.mean() * nv / n_views)})
    return out


def main(split):
    P = {}
    for v in VIEWS:
        p = Path(f"/tmp/predictions_lumpi_{split}_tta_{v}.json")
        if not p.exists():
            print(f"MISSING view {v} ({p})"); return
        P[v] = json.loads(p.read_text())
    fids = sorted(P["id"])
    out = {}
    for fid in fids:
        alld = []
        for v in VIEWS:
            for d in P[v].get(fid, []):
                alld.append({"class": d["class"], "score": d["score"],
                             "box": unflip(d["box"], v), "_v": v})
        out[fid] = fuse(alld)
    o = Path(f"/tmp/predictions_lumpi_{split}_tta_fused.json")
    o.write_text(json.dumps(out))
    n0 = sum(len(v) for v in P["id"].values())
    print(f"{split}: identity {n0} dets -> fused {sum(len(v) for v in out.values())} "
          f"over {len(out)} frames -> {o}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
