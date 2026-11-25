"""
caterpillar_compare.py
======================
Measure the SAME statistics on a CACTUS substrate (from .ply meshes, via
substrate_stats.py) and on a CATERPillar substrate (its output CSV), and print
them side by side, so CATERPillar's settings can be tuned until the two match.

Requirements: substrate_stats.py in the same folder; pip install numpy scipy pandas

Typical use
-----------
    import substrate_stats as ss, caterpillar_compare as cc

    # reference substrate
    cactus = ss.fibres_from_ply(intra_ply, n_theta=32, scale_to_um=1000)
    d_cactus = cc.describe(cactus, box_min_um=(-75, -75, -75), side_um=150,
                           surface="inner", g_ratio=0.5)

    # CATERPillar result (voxel is [0, L]^3 in CATERPillar's own frame!)
    cat = cc.fibres_from_caterpillar_csv("out/run_001.csv", surface="inner")
    d_cat = cc.describe(cat, box_min_um=(0, 0, 0), side_um=150,
                        surface="inner", g_ratio=0.5)

    cc.print_comparison({"CACTUS": d_cactus, "CATERPillar": d_cat})

IMPORTANT: CATERPillar's voxel is [0, L]^3, CACTUS's is centred on 0, so the two
calls use different box_min_um. Use the same `surface` for both.
"""
from __future__ import annotations

import numpy as np

import substrate_stats as ss
from substrate_stats import Fibre


# --------------------------------------------------------------------------- #
# Loader for CATERPillar output
# --------------------------------------------------------------------------- #
def fibres_from_caterpillar_csv(path: str, surface: str = "outer",
                                chunk_rows: int = 2_000_000,
                                max_fibres: int | None = None) -> list[Fibre]:
    """Read the space-delimited CSV written by CATERPillar (<Filename>.csv).

    Only rows with cell_type == 'axon' are used (glia and vessels are ignored).
    Each axon is a run of consecutive rows (spheres). The sphere centres are the
    fibre centreline; the radius column picks the surface:
        surface='inner' -> inner_radius (axon lumen)
        surface='outer' -> outer_radius (myelin outer surface)
    Works on multi-GB files (read in chunks); uses float32 to save memory.
    """
    ss._check_surface(surface)
    import pandas as pd
    rcol = "inner_radius" if surface == "inner" else "outer_radius"
    cols = ["cell_type", "cell_id", "X", "Y", "Z", rcol]
    fibres: list[Fibre] = []
    cur_id, cur_xyz, cur_r = None, [], []

    def flush():
        nonlocal cur_id, cur_xyz, cur_r
        if cur_id is None:
            return
        xyz = np.vstack(cur_xyz).astype(np.float32)
        r = np.concatenate(cur_r).astype(np.float32)
        keep = np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-9]   # drop duplicate seed spheres
        xyz, r = xyz[keep], r[keep]
        if len(xyz) >= 2:
            fibres.append(Fibre(xyz, np.pi * r.astype(np.float64) ** 2))
        cur_id, cur_xyz, cur_r = None, [], []

    for ch in pd.read_csv(path, sep=r"\s+", usecols=cols, chunksize=chunk_rows):
        ch = ch[ch["cell_type"] == "axon"]
        if ch.empty:
            continue
        ids = ch["cell_id"].to_numpy()
        xyz = ch[["X", "Y", "Z"]].to_numpy(np.float64)
        r = ch[rcol].to_numpy(np.float64)
        bounds = np.r_[0, np.flatnonzero(np.diff(ids) != 0) + 1, len(ids)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            gid = int(ids[a])
            if gid != cur_id:
                flush()
                if max_fibres is not None and len(fibres) >= max_fibres:
                    return fibres
                cur_id = gid
            cur_xyz.append(xyz[a:b])
            cur_r.append(r[a:b])
    flush()
    return fibres


# --------------------------------------------------------------------------- #
# Extra statistics
# --------------------------------------------------------------------------- #
def tortuosity_ratio(seg: ss.Segments, min_inside_length_um: float = 20.0) -> dict:
    """Path length / straight-line (chord) length of the part of each fibre inside
    the voxel. 1.0 = straight. Reports mean and median over fibres."""
    m = seg.inside
    path: dict[int, float] = {}
    disp: dict[int, np.ndarray] = {}
    for i, l, v in zip(seg.fibre_id[m], seg.length[m], seg.vec[m]):
        path[i] = path.get(i, 0.0) + l
        disp[i] = disp.get(i, 0.0) + v
    vals = [path[i] / np.linalg.norm(disp[i]) for i in path
            if path[i] >= min_inside_length_um and np.linalg.norm(disp[i]) > 0]
    return {"tortuosity_mean": float(np.mean(vals)), "tortuosity_median": float(np.median(vals))}


def beading_stats(fibres: list[Fibre], box_min_um, side_um: float,
                  lag_um: float = 0.5, min_inside_length_um: float = 20.0) -> dict:
    """Radius variation ALONG each fibre, measured on a common arclength grid so that
    substrates with different node spacing are comparable.

    For every fibre: resample r(s) at spacing `lag_um`, keep points inside the voxel,
    d = r / mean(r) - 1. Pooled over fibres it reports:
        rel_dev_p50/p95/p99 : percentiles of |d|  (-> compare / set BeadingAmplitude ~ p99)
        step_sd             : sd of d(s+lag) - d(s)  (-> relates to BeadingStd)
    CATERPillar clamps r to R*(1 +/- BeadingAmplitude) and takes random steps with sd
    R*BeadingStd per sphere; there is no closed-form mapping to a fixed `lag_um`, so tune
    BeadingStd by running CATERPillar and matching step_sd (and rel_dev) to the target.
    """
    box_min = np.asarray(box_min_um, float)
    dev_all, steps = [], []
    for f in fibres:
        c = f.centre.astype(float)
        seglen = np.linalg.norm(np.diff(c, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seglen)]
        if s[-1] < 2 * lag_um:
            continue
        grid = np.arange(0.0, s[-1], lag_um)
        r = np.interp(grid, s, f.radius.astype(float))
        gx = np.column_stack([np.interp(grid, s, c[:, k]) for k in range(3)])
        ins = np.all((gx >= box_min) & (gx < box_min + side_um), axis=1)
        if ins.sum() * lag_um < min_inside_length_um:
            continue
        d = r / r[ins].mean() - 1.0
        dev_all.append(np.abs(d[ins]))
        both = ins[1:] & ins[:-1]
        steps.append((d[1:] - d[:-1])[both])
    dev = np.concatenate(dev_all)
    st = np.concatenate(steps)
    return {"rel_dev_p50": float(np.percentile(dev, 50)),
            "rel_dev_p95": float(np.percentile(dev, 95)),
            "rel_dev_p99": float(np.percentile(dev, 99)),
            "step_sd": float(st.std()), "beading_lag_um": lag_um}


# --------------------------------------------------------------------------- #
# One call that measures everything, and a side-by-side table
# --------------------------------------------------------------------------- #
def describe(fibres: list[Fibre], box_min_um, side_um: float = 150.0,
             surface: str = "inner", g_ratio: float = 0.5,
             min_inside_length_um: float = 20.0, lag_um: float = 0.5) -> dict:
    """All comparison statistics for one substrate (CACTUS or CATERPillar)."""
    seg = ss.build_segments(fibres, box_min_um, side_um)
    ic = ss.icvf(seg, surface, g_ratio)
    r_in = ss.inner_radii(ss.per_fibre_radius(seg, min_inside_length_um), surface, g_ratio)
    gam = ss.fit_gamma(r_in)
    out = {
        "n_fibres": ic["n_fibres_in_voxel"],
        "icvf_inner": ic["icvf_inner"], "icvf_outer": ic["icvf_outer"],
        "inner_radius_mean_um": gam["observed_mean_um"],
        "gamma_alpha": gam["AlphaMyelin"], "gamma_beta": gam["BetaMyelin"],
        "inner_radius_min_um": gam["min_observed_inner_radius_um"],
        "c2_chord": ss.c2_chord(seg, min_inside_length_um),
        "c2_segments": ss.c2_segments(seg),
        "main_dir_angle_to_z_deg": ss.main_direction(seg)["angle_to_z_deg"],
    }
    out.update(tortuosity_ratio(seg, min_inside_length_um))
    out.update(beading_stats(fibres, box_min_um, side_um, lag_um, min_inside_length_um))
    return out


def print_comparison(descs: dict[str, dict]) -> None:
    """Print a table: rows = statistics, columns = substrates (e.g. CACTUS, CATERPillar)."""
    names = list(descs)
    keys = list(next(iter(descs.values())))
    w = max(len(k) for k in keys) + 2
    print(" " * w + "".join(f"{n:>16}" for n in names))
    for k in keys:
        row = ""
        for n in names:
            v = descs[n][k]
            row += f"{v:>16.4f}" if isinstance(v, float) else f"{v:>16}"
        print(f"{k:<{w}}" + row)