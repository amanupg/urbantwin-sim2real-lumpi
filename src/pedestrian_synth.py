"""Pedestrian synthesizer for UT-LUMPI frames.

Injects ray-cast pedestrians into synthetic frames using the REAL Measurement4
sensor model (extrinsics + per-beam elevation angles from meta.json — valid
because the digital twin shares the real world frame to ~2 cm) and the REAL
train-split placement prior (person_stats.npz).

Method per frame:
  1. Sample N ~ real per-frame count; positions from the placement prior
     (jittered), rejecting spots that collide with existing labeled objects.
  2. Ground height at each spot from the lowest scene points nearby.
  3. For each of the 5 lidars: build a spherical min-depth map of the scene
     (occlusion), then cast the sensor's actual beams (elevation = real beam
     angles, azimuth grid) against an elliptic-capsule person; keep hits that
     are unoccluded; add range noise.
  4. Emit points + a Person label row per placed pedestrian.

Validation: reproduce the real points-per-person-vs-range curve
(person_pointcurve.json) before any training use.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AZ_STEP = np.deg2rad(0.2)      # horizontal resolution
RANGE_NOISE = 0.02             # m, 1 sigma
KEEP_PROB = 0.45               # return dropout, calibrated to real pts/person curve
DEPTH_AZ_BIN = np.deg2rad(0.4)
DEPTH_EL_BIN = np.deg2rad(0.8)
OCCLUSION_MARGIN = 0.6         # m


class SensorModel:
    """Empirical model: origins trilaterated from per-point distance attributes
    (median residual 1 mm), beam elevations measured per ray id. The meta.json
    extrinsics are in a different datum and must not be used for origins."""

    def __init__(self, model_path=ROOT / "tracks/lumpi/sensor_model_empirical.json"):
        model = json.load(open(model_path))
        self.sensors = []
        for sid, s in model.items():
            self.sensors.append({
                "id": sid,
                "origin": np.array(s["origin"], dtype=np.float64),
                "angles": np.array(s["beams"], dtype=np.float64),  # radians, world
            })
        if not self.sensors:
            raise RuntimeError("empty sensor model")


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
    na = int(np.ceil(2 * np.pi / DEPTH_AZ_BIN))
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


def cast_person(sensor, dmap, center, dims, rng_state):
    """Ray-cast one elliptic-capsule person from one sensor. Returns Nx3 pts."""
    o = sensor["origin"]
    l, w, h = dims
    rx, ry = l / 2 * 0.8, w / 2 * 0.8   # capsule slightly inside box
    cz_bottom = center[2] - h / 2
    d = center - o
    dist = np.linalg.norm(d[:2])
    if dist < 1.0 or dist > 120.0:
        return np.zeros((0, 3))
    az_c = np.arctan2(d[1], d[0])
    half_az = np.arctan2(max(rx, ry), dist) * 1.2
    azs = np.arange(az_c - half_az, az_c + half_az, AZ_STEP)
    el_lo = np.arctan2(cz_bottom - o[2], dist)
    el_hi = np.arctan2(cz_bottom + h - o[2], dist)
    els = sensor["angles"][(sensor["angles"] >= el_lo - 0.005) & (sensor["angles"] <= el_hi + 0.005)]
    if len(azs) == 0 or len(els) == 0:
        return np.zeros((0, 3))
    A, E = np.meshgrid(azs, els)
    A, E = A.ravel(), E.ravel()
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1)
    # ray-elliptic-cylinder intersection in world frame (axis vertical at center)
    ox, oy = o[0] - center[0], o[1] - center[1]
    dx, dy = dirs[:, 0], dirs[:, 1]
    a = (dx / rx) ** 2 + (dy / ry) ** 2
    b = 2 * (ox * dx / rx ** 2 + oy * dy / ry ** 2)
    c = (ox / rx) ** 2 + (oy / ry) ** 2 - 1
    disc = b ** 2 - 4 * a * c
    ok = disc > 0
    t = np.full(len(dirs), np.inf)
    t[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    zhit = o[2] + t * dirs[:, 2]
    ok &= (t > 0) & (zhit >= cz_bottom) & (zhit <= cz_bottom + h)
    if not ok.any():
        return np.zeros((0, 3))
    t = t[ok]; dirs = dirs[ok]; A = A[ok]; E = E[ok]
    # occlusion: scene depth along this ray must not be clearly in front
    keep = np.array([
        _depth_at(dmap, a_, e_) > (t_ - OCCLUSION_MARGIN)
        for a_, e_, t_ in zip(A, E, t)
    ])
    t = t[keep]; dirs = dirs[keep]
    drop = rng_state.random(len(t)) < KEEP_PROB
    t = t[drop]; dirs = dirs[drop]
    t = t + rng_state.normal(0, RANGE_NOISE, len(t))
    return o + t[:, None] * dirs


def ground_z(scene, x, y, fallback=-2.1):
    m = (np.abs(scene[:, 0] - x) < 1.2) & (np.abs(scene[:, 1] - y) < 1.2)
    if m.sum() < 5:
        return fallback
    return float(np.percentile(scene[m, 2], 5))


def synthesize_frame(scene_pts, labels, sensors, prior_pos, prior_dims, rng,
                     n_persons=None, group_frac=0.30):
    """Returns (new_points list, new_label_rows list).

    v2: ~30% of placements are GROUPS (2-4 people within ~1m of a shared
    spot) — real pedestrians cluster; run1-v1 placed everyone independently.
    """
    if n_persons is None:
        n_persons = int(np.clip(rng.poisson(22), 4, 58))
    # expand into groups: list of (x_seed, size) clusters
    clusters = []
    while len(clusters) < n_persons:
        if rng.random() < group_frac and len(clusters) < n_persons - 1:
            clusters.append((None, int(rng.integers(2, 5))))
        else:
            clusters.append((None, 1))
    # existing object footprints for collision rejection
    boxes = [(float(f[0]), float(f[1]), max(float(f[3]), float(f[4])) / 2 + 0.6)
             for f in (ln.split() for ln in labels) if len(f) >= 8]
    dmaps = None  # lazy per-sensor depth maps
    out_pts, out_rows = [], []
    tries = 0
    ci = 0
    while ci < len(clusters) and tries < n_persons * 8:
        tries += 1
        _, size = clusters[ci]
        k = int(rng.integers(len(prior_pos)))
        gx, gy = prior_pos[k] + rng.normal(0, 0.8, 2)
        members = []
        for _ in range(size):
            x, y = gx + rng.normal(0, 0.6), gy + rng.normal(0, 0.6)
            if any((x - bx) ** 2 + (y - by) ** 2 < br ** 2 for bx, by, br in boxes):
                continue
            dims = prior_dims[int(rng.integers(len(prior_dims)))] * rng.normal(1.0, 0.05, 3)
            gz = ground_z(scene_pts, x, y)
            center = np.array([x, y, gz + dims[2] / 2])
            if dmaps is None:
                dmaps = [_depth_map(scene_pts, s["origin"]) for s in sensors]
            pts = [cast_person(s, dm, center, dims, rng)
                   for s, dm in zip(sensors, dmaps)]
            pts = np.concatenate([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 3))
            if len(pts) < 5:
                continue
            heading = float(rng.uniform(-np.pi, np.pi))
            members.append((pts.astype(np.float32),
                            f"{x:.4f} {y:.4f} {center[2]:.4f} "
                            f"{dims[0]:.3f} {dims[1]:.3f} {dims[2]:.3f} {heading:.4f} Person 0"))
            boxes.append((x, y, 0.8))
        if members:
            for p, r in members:
                out_pts.append(p); out_rows.append(r)
            ci += 1
    return out_pts, out_rows
