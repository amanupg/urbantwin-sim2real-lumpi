"""Bike+rider composite synthesizer for UT-LUMPI frames (run5 ingredient).

Why: real bicycles/motorcycles in LUMPI carry a rider (2.0x0.84x1.9 m), but
UT-LUMPI synthetic bikes are riderless (1.28m) -- the model calls real bikes
"Person" (419/808 confusions measured on the late slice) or misses them.
This synthesizer ray-casts a COMPOSITE object: an ellipsoid bike frame
(semi-axes along the heading) + the proven elliptic rider capsule on top,
against the real 5-sensor LUMPI model, with the same occlusion depth maps,
dropout calibration, and range noise as pedestrian_synth.py.

Method per frame:
  1. Sample counts ~ real per-frame distribution (bike_stats.npz: mean ~4.9
     bicycles/frame, motorcycles rare ~1 per 20 frames).
  2. Place from the real position prior (jittered), ground from local scene,
     reject collisions with labeled objects.
  3. Cast both parts (ellipsoid bike + rider capsule) from all 5 sensors;
     keep unoccluded hits; dropout KEEP_PROB calibrated to the real
     points-per-bike-vs-range curve (bike_pointcurve.json).

Validation gate: synth/real pts-per-bike per range bin before any training use.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AZ_STEP = np.deg2rad(0.2)
RANGE_NOISE = 0.02
KEEP_PROB = (0.60, 0.55, 15.0)  # (near <15m, far >=15m, split): calibrated
DEPTH_AZ_BIN = np.deg2rad(0.4)
DEPTH_EL_BIN = np.deg2rad(0.8)
OCCLUSION_MARGIN = 0.6

RIDER_DIMS = (0.50, 0.42, 1.65)   # rider capsule l,w,h (m)
BIKE_SEMI = (1.05, 0.33, 0.55)    # ellipsoid semi-axes: along-heading, across, up
BIKE_CENTER_H = 0.55              # bike ellipsoid center height above ground
RIDER_BOTTOM_H = 0.75             # rider capsule bottom above ground (seat)


def _spherical(pts, origin):
    d = pts - origin
    rng = np.linalg.norm(d, axis=1)
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / np.maximum(rng, 1e-9), -1, 1))
    return az, el, rng


def _depth_map(pts, origin):
    az, el, rng = _spherical(pts, origin)
    ai = np.floor((az + np.pi) / DEPTH_AZ_BIN).astype(np.int64)
    ei = np.floor((el + np.pi / 2) / DEPTH_EL_BIN).astype(np.int64)
    key = ai * 1000 + ei
    order = np.argsort(key, kind="stable")
    key_s, rng_s = key[order], rng[order]
    first = np.ones(len(key_s), dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    idx = np.flatnonzero(first)
    dmap = {}
    for i, start in enumerate(idx):
        end = idx[i + 1] if i + 1 < len(idx) else len(key_s)
        dmap[int(key_s[start])] = float(rng_s[start:end].min())
    return dmap


def _depth_at(dmap, az, el):
    ai = int((az + np.pi) / DEPTH_AZ_BIN)
    ei = int((el + np.pi / 2) / DEPTH_EL_BIN)
    return dmap.get(ai * 1000 + ei, np.inf)


def _rays_to_hits(sensor, dmap, center, semi, heading, h_lo, h_hi, rng_state, keep_prob, ellipsoid=True):
    """Cast a part (ellipsoid frame or vertical capsule rider) and return Nx3."""
    o = sensor["origin"]
    dist = np.hypot(center[0] - o[0], center[1] - o[1])
    if dist < 1.0 or dist > 120.0:
        return np.zeros((0, 3))
    az_c = np.arctan2(center[1] - o[1], center[0] - o[0])
    half_az = np.arctan2(max(semi[0], semi[1]), dist) * 1.25
    azs = np.arange(az_c - half_az, az_c + half_az, AZ_STEP)
    el_lo = np.arctan2(center[2] + h_lo - o[2], dist)
    el_hi = np.arctan2(center[2] + h_hi - o[2], dist)
    els = sensor["angles"][(sensor["angles"] >= el_lo - 0.005) & (sensor["angles"] <= el_hi + 0.005)]
    if len(azs) == 0 or len(els) == 0:
        return np.zeros((0, 3))
    A, E = np.meshgrid(azs, els)
    A, E = A.ravel(), E.ravel()
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1)
    # ray-ellipsoid intersection in the part's local frame
    ch, sh = np.cos(-heading), np.sin(-heading)
    rel = o - center
    ox, oy = rel[0] * ch - rel[1] * sh, rel[0] * sh + rel[1] * ch
    oz = rel[2]
    dx = dirs[:, 0] * ch - dirs[:, 1] * sh
    dy = dirs[:, 0] * sh + dirs[:, 1] * ch
    dz = dirs[:, 2]
    a, b, c_ = semi
    A2 = (dx / a) ** 2 + (dy / b) ** 2 + (dz / c_) ** 2
    B2 = 2 * (ox * dx / a ** 2 + oy * dy / b ** 2 + oz * dz / c_ ** 2)
    C2 = (ox / a) ** 2 + (oy / b) ** 2 + (oz / c_) ** 2 - 1
    disc = B2 ** 2 - 4 * A2 * C2
    ok = disc > 0
    t = np.full(len(dirs), np.inf)
    t[ok] = (-B2[ok] - np.sqrt(disc[ok])) / (2 * A2[ok])
    ok &= t > 0
    if not ok.any():
        return np.zeros((0, 3))
    t = t[ok]; A = A[ok]; E = E[ok]
    keep = np.array([
        _depth_at(dmap, a_, e_) > (t_ - OCCLUSION_MARGIN)
        for a_, e_, t_ in zip(A, E, t)
    ])
    t = t[keep]
    dirs_out = dirs[ok][keep]
    if isinstance(keep_prob, tuple):
        lo_kp, hi_kp, split = keep_prob
        probs = np.where(t < split, lo_kp, hi_kp)
        drop = rng_state.random(len(t)) < probs
    else:
        drop = rng_state.random(len(t)) < keep_prob
    t = t[drop]; dirs_out = dirs_out[drop]
    t = t + rng_state.normal(0, RANGE_NOISE, len(t))
    return o + t[:, None] * dirs_out


def cast_composite(sensor, dmap, center, heading, rng, keep_prob):
    """Bike ellipsoid (along heading) + rider capsule (vertical). Returns Nx3."""
    pts = []
    bike_c = np.array([center[0], center[1], center[2] + BIKE_CENTER_H])
    pts.append(_rays_to_hits(sensor, dmap, bike_c, BIKE_SEMI, heading,
                             -BIKE_SEMI[2], BIKE_SEMI[2], rng, keep_prob))
    rl, rw, rh = RIDER_DIMS
    rider_c = np.array([center[0], center[1], center[2] + RIDER_BOTTOM_H + rh / 2])
    pts.append(_rays_to_hits(sensor, dmap, rider_c, (rl / 2 * 0.8, rw / 2 * 0.8, rh / 2), 0.0,
                             -rh / 2, rh / 2, rng, keep_prob))
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def ground_z(scene, x, y):
    m = (np.abs(scene[:, 0] - x) < 1.2) & (np.abs(scene[:, 1] - y) < 1.2)
    if m.sum() < 5:
        return None
    return float(np.percentile(scene[m, 2], 5))


def synthesize_bikes(scene_pts, labels, sensors, prior, counts_dist, rng,
                     n_bikes=None, n_motorcycles=None, keep_prob=KEEP_PROB):
    """Returns (new_points list, new_label_rows list)."""
    pos, dims = prior["pos"], prior["dims"]
    if n_bikes is None:
        n_bikes = int(np.clip(rng.poisson(4.9), 1, 12))
    if n_motorcycles is None:
        n_motorcycles = int(rng.random() < 0.08)
    boxes = [(float(f[0]), float(f[1]), max(float(f[3]), float(f[4])) / 2 + 0.6)
             for f in (ln.split() for ln in labels) if len(f) >= 8]
    dmaps = None
    out_pts, out_rows = [], []
    total = n_bikes + n_motorcycles
    tries = 0
    placed = 0
    while placed < total and tries < total * 8:
        tries += 1
        k = int(rng.integers(len(pos)))
        x, y = pos[k] + rng.normal(0, 0.8, 2)
        if not (-39.0 < x < 39.0 and -39.0 < y < 39.0):
            continue
        if any((x - bx) ** 2 + (y - by) ** 2 < br ** 2 for bx, by, br in boxes):
            continue
        gz = ground_z(scene_pts, x, y)
        if gz is None or not (-10.0 < gz < 2.0):
            continue
        is_moto = placed >= n_bikes
        cls = "Motorcycle" if is_moto else "Bicycle"
        dd = dims[int(rng.integers(len(dims)))] * rng.normal(1.0, 0.05, 3)
        heading = float(rng.uniform(-np.pi, np.pi))
        if dmaps is None:
            dmaps = [_depth_map(scene_pts, s["origin"]) for s in sensors]
        center = np.array([x, y, gz])
        pts = np.concatenate([cast_composite(s, dm, center, heading, rng, keep_prob)
                              for s, dm in zip(sensors, dmaps)]) if sensors else np.zeros((0, 3))
        pts = pts if len(pts) else np.zeros((0, 3))
        if len(pts) < 8:
            continue
        out_pts.append(pts.astype(np.float32))
        out_rows.append(f"{x:.4f} {y:.4f} {gz + dd[2]/2:.4f} "
                        f"{dd[0]:.3f} {dd[1]:.3f} {dd[2]:.3f} {heading:.4f} {cls} 0")
        boxes.append((x, y, 1.2))
        placed += 1
    return out_pts, out_rows
