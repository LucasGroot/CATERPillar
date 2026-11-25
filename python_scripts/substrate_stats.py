"""
substrate_stats.py
==================
Compute the numbers needed to set up a comparable healthy CATERPillar substrate
from tube-shaped fibre meshes (.ply):

    * ICVF of the fibres inside the voxel          -> AxonsWithMyelinICVF (in PERCENT)
    * inner-radius Gamma fit                       -> AlphaMyelin, BetaMyelin
    * smallest inner radius                        -> MinRadius
    * orientation dispersion c2 = <cos^2(psi)>     -> FODF_c2
    * constant g-ratio                             -> K1, K2, K3

Requirements:  pip install numpy scipy plyfile

ASSUMPTION ABOUT THE MESH LAYOUT (check it with inspect_ply() first)
--------------------------------------------------------------------
Each fibre is a connected tube mesh whose vertices are stored ring by ring:
`n_theta` consecutive vertices form one cross-sectional ring (n_theta = 32 in
the file). Up to two extra trailing vertices per fibre (e.g. end-cap
centres) are tolerated. If your files are laid out differently, build the
fibres yourself from centreline points and radii with `Fibre.from_centreline`.

UNITS
-----
Everything is converted to micrometres on reading (the .ply is assumed to be in mm, so
scale_to_um=1000). All outputs are in micrometres / fractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Fibre:
    """One fibre as an ordered list of cross-sections.

    centre : (N, 3) float, ring centres in um (ordered along the fibre)
    area   : (N,)   float, cross-section area of the *meshed surface* in um^2
    """
    centre: np.ndarray
    area: np.ndarray

    @classmethod
    def from_centreline(cls, centre_um: np.ndarray, radius_um: np.ndarray) -> "Fibre":
        """Build a fibre from centreline points (N,3) in um and radii (N,) in um.
        The radius must be the radius of the SAME surface you declare in
        `surface=` later ('inner' = axon, 'outer' = myelin outer surface)."""
        centre_um = np.asarray(centre_um, float)
        radius_um = np.asarray(radius_um, float)
        if centre_um.ndim != 2 or centre_um.shape[1] != 3 or len(centre_um) != len(radius_um):
            raise ValueError("centre_um must be (N,3) and radius_um (N,)")
        return cls(centre_um, np.pi * radius_um ** 2)

    @property
    def radius(self) -> np.ndarray:
        return np.sqrt(self.area / np.pi)


@dataclass
class Segments:
    """All fibre segments, flattened. 'inside' marks segments whose midpoint
    lies in the voxel (these are the only ones used for statistics)."""
    fibre_id: np.ndarray   # (S,)
    vec: np.ndarray        # (S,3) segment vector, um
    length: np.ndarray     # (S,)  um
    area: np.ndarray       # (S,)  mean cross-section area, um^2
    inside: np.ndarray     # (S,)  bool
    side_um: float
    box_min_um: np.ndarray


# --------------------------------------------------------------------------- #
# Reading PLY and turning meshes into fibres
# --------------------------------------------------------------------------- #
def read_ply(path: str, scale_to_um: float = 1000.0, rotation: np.ndarray | None = None):
    """Read vertices (V,3) in um and triangle faces (F,3) (or None).

    path        : .ply file (ascii or binary)
    scale_to_um : factor to micrometres. 1000 if the file is in mm.
    rotation    : optional 3x3 matrix R; vertices become  v @ R.T.
                  Use it only if the file is not already in the frame where the
                  main fibre direction is z (check with main_direction()).
    """
    out = _read_ply_fast(path)
    if out is None:                                  # ascii / unusual layout
        out = _read_ply_plyfile(path)
    v, faces = out
    v = v * np.float32(scale_to_um) if v.dtype == np.float32 else v * scale_to_um
    if rotation is not None:
        v = (v.astype(float) @ np.asarray(rotation, float).T).astype(v.dtype)
    return v, faces


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
              "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def _parse_ply_header(path: str):
    with open(path, "rb") as fh:
        lines = []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            lines.append(line)
            if line == "end_header" or not line and len(lines) > 200:
                break
        offset = fh.tell()
    fmt = next(l.split()[1] for l in lines if l.startswith("format"))
    elements = []
    for l in lines:
        t = l.split()
        if l.startswith("element"):
            elements.append([t[1], int(t[2]), []])
        elif l.startswith("property") and elements:
            elements[-1][2].append(t[1:])
    return lines, fmt, elements, offset


def peek_ply(path: str) -> None:
    """Print the header only (no data loaded): element counts and file size.
    Run this first on a big file."""
    lines, fmt, elements, _ = _parse_ply_header(path)
    import os
    print(f"{os.path.getsize(path) / 1e6:.1f} MB, format {fmt}")
    for name, count, props in elements:
        print(f"element {name}: {count} ; properties: {[' '.join(p) for p in props]}")


def _read_ply_fast(path: str):
    """Memory-lean reader for binary little-endian PLY with
    vertex (scalar props, incl. x y z) then triangle faces (one list property).
    Returns (vertices float32/64 (V,3), faces int32 (F,3)) or None if unsupported."""
    lines, fmt, elements, offset = _parse_ply_header(path)
    if fmt != "binary_little_endian" or len(elements) < 1 or elements[0][0] != "vertex":
        return None
    vprops = elements[0][2]
    if any(p[0] == "list" for p in vprops):
        return None
    vdt = np.dtype([(p[1], _PLY_TYPES[p[0]]) for p in vprops])
    nv = elements[0][1]
    vert = np.fromfile(path, dtype=vdt, count=nv, offset=offset)
    v = np.column_stack([vert["x"], vert["y"], vert["z"]])
    del vert
    faces = None
    if len(elements) > 1 and elements[1][0] == "face":
        fprops = elements[1][2]
        if len(fprops) != 1 or fprops[0][0] != "list":
            return None
        cnt_t, item_t = _PLY_TYPES[fprops[0][1]], _PLY_TYPES[fprops[0][2]]
        nf = elements[1][1]
        fdt = np.dtype([("n", cnt_t), ("i", item_t, (3,))])
        raw = np.fromfile(path, dtype=fdt, count=nf, offset=offset + nv * vdt.itemsize)
        if len(raw) != nf or not np.all(raw["n"] == 3):
            return None                              # not pure triangles -> fallback
        faces = raw["i"].astype(np.int32)
        del raw
    return v, faces


def _read_ply_plyfile(path: str):
    """Fallback reader (ascii or unusual layouts). Heavier on memory."""
    from plyfile import PlyData
    ply = PlyData.read(path)
    vx = ply["vertex"]
    v = np.column_stack([np.asarray(vx[k]) for k in ("x", "y", "z")])
    faces = None
    if "face" in ply:
        names = ply["face"].data.dtype.names
        key = "vertex_indices" if "vertex_indices" in names else names[0]
        fa = ply["face"][key]
        lens = np.fromiter((len(f) for f in fa), int, count=len(fa))
        if np.all(lens == 3):
            faces = np.concatenate(fa).reshape(-1, 3).astype(np.int32)
        else:
            tri = []
            for f in fa:
                for k in range(1, len(f) - 1):       # fan-triangulate polygons
                    tri.append((f[0], f[k], f[k + 1]))
            faces = np.asarray(tri, np.int32)
    return v, faces


def split_components(n_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    """Vertex-index arrays of each fibre, assuming each fibre's vertices are stored
    contiguously (true for ring-by-ring files written fibre after fibre).
    A boundary is placed wherever no face spans two neighbouring vertex indices.
    Uses almost no memory. Wrong splits are caught by the ring-layout check."""
    lo = faces.min(axis=1).astype(np.int64)
    hi = faces.max(axis=1).astype(np.int64)
    cover = np.bincount(lo + 1, minlength=n_vertices + 2) - np.bincount(hi + 1, minlength=n_vertices + 2)
    cover = np.cumsum(cover)[: n_vertices + 1]       # cover[p] > 0: some face spans p-1 and p
    starts = np.concatenate([[0], np.where(cover[1:n_vertices] <= 0)[0] + 1])
    ends = np.concatenate([starts[1:], [n_vertices]])
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def split_components_graph(n_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    """Slow, memory-hungry general version (true connected components). Only use on
    small files, or if fibre vertices are NOT stored contiguously."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    g = coo_matrix((np.ones(2 * len(a), dtype=np.int8),
                    (np.concatenate([a, b]), np.concatenate([b, c]))),
                   shape=(n_vertices, n_vertices))
    n, lab = connected_components(g, directed=False)
    return [np.where(lab == k)[0] for k in range(n)]


def _rings_from_vertices(vs: np.ndarray, n_theta: int, tol: float = 0.25):
    """Reshape one fibre's vertices into (N, n_theta, 3) rings, validating the layout."""
    n = len(vs)
    for extra in (0, 1, 2):                      # tolerate trailing cap-centre vertices
        m = n - extra
        if m <= 0 or m % n_theta:
            continue
        rings = vs[:m].reshape(-1, n_theta, 3)
        c = rings.mean(axis=1)
        d = np.linalg.norm(rings - c[:, None, :], axis=2)
        rel = d.std(axis=1) / np.maximum(d.mean(axis=1), 1e-12)
        if np.mean(rel < tol) >= 0.95:
            return rings
    raise ValueError(
        f"Cannot read a fibre with {n} vertices as rings of {n_theta}. "
        "Either n_theta is wrong or vertices are not stored ring by ring. "
        "Use Fibre.from_centreline instead.")


def fibres_from_rings(rings: np.ndarray) -> Fibre:
    """rings: (N, n_theta, 3) in um -> Fibre (centre and polygon area per ring)."""
    c = rings.mean(axis=1)
    pc = rings - c[:, None, :]
    cross = np.cross(pc, np.roll(pc, -1, axis=1))
    area = np.linalg.norm(0.5 * cross.sum(axis=1), axis=1)
    return Fibre(c, area)


def _open_streaming(path: str):
    """Memory-map a binary little-endian triangle PLY. Returns (vmm, fmm, nv, nf)
    or None if the layout is not supported (then the in-memory path is used)."""
    lines, fmt, elements, offset = _parse_ply_header(path)
    if fmt != "binary_little_endian" or len(elements) < 2:
        return None
    if elements[0][0] != "vertex" or elements[1][0] != "face":
        return None
    vprops, fprops = elements[0][2], elements[1][2]
    if any(p[0] == "list" for p in vprops) or len(fprops) != 1 or fprops[0][0] != "list":
        return None
    vdt = np.dtype([(p[1], _PLY_TYPES[p[0]]) for p in vprops])
    fdt = np.dtype([("n", _PLY_TYPES[fprops[0][1]]), ("i", _PLY_TYPES[fprops[0][2]], (3,))])
    nv, nf = elements[0][1], elements[1][1]
    vmm = np.memmap(path, dtype=vdt, mode="r", offset=offset, shape=(nv,))
    fmm = np.memmap(path, dtype=fdt, mode="r", offset=offset + nv * vdt.itemsize, shape=(nf,))
    return vmm, fmm, nv, nf


def _fibre_ranges_streaming(fmm, nv: int, chunk_faces: int = 4_000_000):
    """Split vertices into fibres by streaming through the faces in chunks.
    Assumes each fibre's vertices are stored contiguously. Memory: ~4 bytes/vertex."""
    cover = np.zeros(nv + 2, dtype=np.int32)
    nf = len(fmm)
    for a in range(0, nf, chunk_faces):
        blk = np.asarray(fmm[a:a + chunk_faces])
        if not np.all(blk["n"] == 3):
            raise ValueError("Non-triangle faces found; streaming reader needs triangles.")
        idx = blk["i"].astype(np.int64)
        lo, hi = idx.min(axis=1), idx.max(axis=1)
        base = int(lo.min())
        span = int(hi.max()) + 2 - base
        delta = np.bincount(lo + 1 - base, minlength=span) - np.bincount(hi + 1 - base, minlength=span)
        cover[base:base + span] += delta.astype(np.int32)
    np.cumsum(cover, out=cover)
    starts = np.concatenate([[0], np.where(cover[1:nv] <= 0)[0] + 1])
    ends = np.concatenate([starts[1:], [nv]])
    return starts, ends


def bounding_box_um(path: str, scale_to_um: float = 1000.0, chunk: int = 5_000_000):
    """Min and max corner (um) of all vertices, read in chunks (works on huge files).
    Use it to choose box_min_um."""
    st = _open_streaming(path)
    if st is None:
        v, _ = read_ply(path, scale_to_um)
        return v.min(0), v.max(0)
    vmm, _, nv, _ = st
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for a in range(0, nv, chunk):
        b = np.asarray(vmm[a:a + chunk])
        xyz = np.column_stack([b["x"], b["y"], b["z"]]).astype(float)
        lo = np.minimum(lo, xyz.min(0))
        hi = np.maximum(hi, xyz.max(0))
    return lo * scale_to_um, hi * scale_to_um


def fibres_from_ply(path: str, n_theta: int = 32, scale_to_um: float = 1000.0,
                    rotation: np.ndarray | None = None,
                    max_fibres: int | None = None, verbose: bool = True,
                    skip_bad: bool = False) -> list[Fibre]:
    """Read one .ply that contains one or many fibre tubes, fibre by fibre.

    Works on multi-GB files: only a few hundred MB of RAM are used, because the
    file is memory-mapped and only the per-ring centres and areas are kept.
    max_fibres : only process the first K fibres (quick test).
    skip_bad   : skip fibres that fail the ring-layout check instead of raising.
    """
    rot = None if rotation is None else np.asarray(rotation, float)
    st = _open_streaming(path)
    out: list[Fibre] = []
    n_bad = 0

    def _add(vs_um: np.ndarray):
        nonlocal n_bad
        try:
            rings = _rings_from_vertices(vs_um, n_theta)
        except ValueError:
            if skip_bad:
                n_bad += 1
                return
            raise
        if rot is not None:
            rings = rings @ rot.T
        out.append(fibres_from_rings(rings))

    if st is None:                                   # ascii etc.: in-memory path
        v, faces = read_ply(path, scale_to_um)
        if faces is None:
            raise ValueError("No faces in file: cannot separate fibres. "
                             "Build Fibre objects with Fibre.from_centreline.")
        for idx in split_components(len(v), faces):
            if max_fibres is not None and len(out) >= max_fibres:
                break
            _add(v[idx].astype(float))
        return out

    vmm, fmm, nv, nf = st
    if verbose:
        print(f"streaming {nv} vertices / {nf} faces ...", flush=True)
    starts, ends = _fibre_ranges_streaming(fmm, nv)
    if verbose:
        print(f"{len(starts)} fibres found", flush=True)
    for s, e in zip(starts, ends):
        if max_fibres is not None and len(out) >= max_fibres:
            break
        b = np.asarray(vmm[s:e])
        xyz = np.column_stack([b["x"], b["y"], b["z"]]).astype(float) * scale_to_um
        _add(xyz)
    if verbose and n_bad:
        print(f"skipped {n_bad} fibres that failed the ring-layout check")
    return out


def inspect_ply(path: str, n_theta: int = 32, scale_to_um: float = 1000.0,
                max_fibres: int | None = 300) -> None:
    """Run this FIRST. Prints the bounding box, the number of fibres, and checks the
    ring layout on the first `max_fibres` fibres (None = all). Safe on huge files."""
    peek_ply(path)
    lo, hi = bounding_box_um(path, scale_to_um)
    print("bounding box (um): min", lo.round(2), " max", hi.round(2))
    st = _open_streaming(path)
    if st is None:
        print("(not a binary little-endian triangle PLY: using the slow in-memory reader)")
    fibres = fibres_from_ply(path, n_theta, scale_to_um, max_fibres=None if st is None else max_fibres,
                             skip_bad=True)
    print(f"fibres that passed the ring-layout check: {len(fibres)}"
          f"{'' if max_fibres is None else f' (first {max_fibres} tried)'}")
    if fibres:
        n_rings = np.array([len(f.centre) for f in fibres])
        r = np.concatenate([f.radius for f in fibres])
        print("rings per fibre: min %d  median %d  max %d" % (n_rings.min(), np.median(n_rings), n_rings.max()))
        print("ring radius (um): min %.3f  median %.3f  max %.3f" % (r.min(), np.median(r), r.max()))


# --------------------------------------------------------------------------- #
# Voxel handling
# --------------------------------------------------------------------------- #
def build_segments(fibres: Sequence[Fibre], box_min_um: Sequence[float], side_um: float = 150.0) -> Segments:
    """Flatten fibres into segments and flag those inside the cubic voxel
    [box_min, box_min + side]^3 (decided by the segment midpoint).

    box_min_um : (3,) the voxel's minimum corner in um, in the same frame as the
                 fibres. You must know this; if the voxel is centred on the
                 origin use (-side/2,)*3, if it starts at the origin use (0,0,0).
    """
    box_min = np.asarray(box_min_um, float)
    ids, vec, ln, ar, ins = [], [], [], [], []
    for i, f in enumerate(fibres):
        d = np.diff(f.centre, axis=0)
        mid = 0.5 * (f.centre[:-1] + f.centre[1:])
        ids.append(np.full(len(d), i))
        vec.append(d)
        ln.append(np.linalg.norm(d, axis=1))
        ar.append(0.5 * (f.area[:-1] + f.area[1:]))
        ins.append(np.all((mid >= box_min) & (mid < box_min + side_um), axis=1))
    inside = np.concatenate(ins)
    if not inside.any():
        allc = np.vstack([f.centre for f in fibres])
        raise ValueError(
            "No fibre segments lie inside the voxel. Check box_min_um, side_um and the units "
            f"(scale_to_um). The fibre centres span min {allc.min(0).round(1)} to max "
            f"{allc.max(0).round(1)} um; the voxel you gave is {box_min.round(1)} to "
            f"{(box_min + side_um).round(1)} um.")
    return Segments(np.concatenate(ids), np.vstack(vec), np.concatenate(ln),
                    np.concatenate(ar), inside, float(side_um), box_min)


def _fibre_inside_length(seg: Segments) -> dict[int, float]:
    m = seg.inside
    out: dict[int, float] = {}
    for i, l in zip(seg.fibre_id[m], seg.length[m]):
        out[i] = out.get(i, 0.0) + l
    return out


# --------------------------------------------------------------------------- #
# The four quantities
# --------------------------------------------------------------------------- #
def icvf(seg: Segments, surface: str = "outer", g_ratio: float = 0.5) -> dict:
    """Volume fraction of fibres inside the voxel.

    surface : which surface your meshes represent: 'outer' (myelin outer surface)
              or 'inner' (axon). Decides the conversion between the two.
    g_ratio : inner/outer radius ratio. Area scales as g^2.

    Returns fractions (multiply by 100 for the JSON). CATERPillar's
    AxonsWithMyelinICVF is the *outer* one.
    """
    _check_surface(surface)
    m = seg.inside
    frac = float(np.sum(seg.area[m] * seg.length[m]) / seg.side_um ** 3)
    if surface == "outer":
        outer, inner = frac, frac * g_ratio ** 2
    else:
        inner, outer = frac, frac / g_ratio ** 2
    return {"icvf_inner": inner, "icvf_outer": outer,
            "AxonsWithMyelinICVF_percent": 100.0 * outer,
            "n_fibres_in_voxel": len(_fibre_inside_length(seg))}


def per_fibre_radius(seg: Segments, min_inside_length_um: float = 20.0) -> np.ndarray:
    """One effective radius per fibre (um) of the meshed surface:
    sqrt( sum(A*L) / (pi * sum(L)) ) over the part of the fibre inside the voxel.

    Fibres with less than `min_inside_length_um` inside the voxel are dropped
    (their radius is dominated by edge effects). 20 um is a judgement call;
    check that the result does not change much with 10 or 40.
    """
    m = seg.inside
    num: dict[int, float] = {}
    den: dict[int, float] = {}
    for i, a, l in zip(seg.fibre_id[m], seg.area[m], seg.length[m]):
        num[i] = num.get(i, 0.0) + a * l
        den[i] = den.get(i, 0.0) + l
    r = [np.sqrt(num[i] / (np.pi * den[i])) for i in den if den[i] >= min_inside_length_um]
    return np.asarray(r)


def inner_radii(r_surface: np.ndarray, surface: str = "outer", g_ratio: float = 0.5) -> np.ndarray:
    """Convert per-fibre radii of the meshed surface to inner (axon) radii."""
    _check_surface(surface)
    return np.asarray(r_surface) * (g_ratio if surface == "outer" else 1.0)


def fit_gamma(r_inner_um: np.ndarray) -> dict:
    """Maximum-likelihood Gamma fit (location fixed at 0) to inner radii in um.

    CATERPillar draws each myelinated axon's inner radius from
    Gamma(shape=AlphaMyelin, scale=BetaMyelin) and rejects draws <= MinRadius.
    Returns AlphaMyelin, BetaMyelin, the fitted mean, a KS p-value, and the
    smallest observed radius (use at or just below it for MinRadius).
    Needs >= ~30 fibres to be meaningful.
    """
    r = np.asarray(r_inner_um, float)
    if len(r) < 30:
        print(f"WARNING: only {len(r)} fibres; the Gamma fit will be unreliable.")
    a, _, scale = stats.gamma.fit(r, floc=0)
    ks = stats.kstest(r, "gamma", args=(a, 0, scale))
    return {"AlphaMyelin": float(a), "BetaMyelin": float(scale),
            "fitted_mean_um": float(a * scale), "observed_mean_um": float(r.mean()),
            "ks_pvalue": float(ks.pvalue), "n": int(len(r)),
            "min_observed_inner_radius_um": float(r.min())}


def c2_chord(seg: Segments, min_inside_length_um: float = 20.0) -> float:
    """<cos^2(psi)> the way CATERPillar reports it in <Filename>_growth_info.txt
    (cosPhiSquared): for each fibre, psi is the angle between z and the straight
    line from the fibre's first to last point (here: net displacement of the part
    inside the voxel); the plain average over fibres. Use THIS to set FODF_c2."""
    m = seg.inside
    disp: dict[int, np.ndarray] = {}
    for i, v in zip(seg.fibre_id[m], seg.vec[m]):
        disp[i] = disp.get(i, 0.0) + v
    inside_len = _fibre_inside_length(seg)
    vals = []
    for i, d in disp.items():
        n = np.linalg.norm(d)
        if inside_len[i] >= min_inside_length_um and n > 0:
            vals.append((d[2] / n) ** 2)
    return float(np.mean(vals))


def c2_segments(seg: Segments) -> float:
    """Length-weighted <cos^2(psi)> of the LOCAL tangents. Always <= c2_chord for
    wavy fibres. Report it as a morphology check, not for FODF_c2."""
    m = seg.inside
    cos2 = (seg.vec[m, 2] / np.maximum(seg.length[m], 1e-12)) ** 2
    return float(np.sum(cos2 * seg.length[m]) / np.sum(seg.length[m]))


def main_direction(seg: Segments) -> dict:
    """Dominant fibre direction: largest eigenvector of the length-weighted
    orientation tensor. angle_to_z_deg should be small (a few degrees) if the
    mesh is in the frame where the main direction is z. If not, pass the proper
    `rotation` to read_ply."""
    m = seg.inside
    t = seg.vec[m] / np.maximum(seg.length[m], 1e-12)[:, None]
    T = (t.T * seg.length[m]) @ t / seg.length[m].sum()
    w, V = np.linalg.eigh(T)
    d = V[:, -1]
    ang = np.degrees(np.arccos(min(1.0, abs(d[2]))))
    return {"direction": d, "angle_to_z_deg": float(ang), "eigenvalues": w[::-1]}


def myelin_K_from_g_ratio(g_ratio: float) -> dict:
    """K1, K2, K3 for a CONSTANT g-ratio in CATERPillar.
    thickness = K1 + K2*D_in + K3*ln(D_in)  and  thickness = r_in*(1/g - 1)
    =>  K1 = 0, K2 = (1/g - 1)/2, K3 = 0."""
    return {"K1": 0.0, "K2": (1.0 / g_ratio - 1.0) / 2.0, "K3": 0.0}


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #
def caterpillar_values(fibres: Sequence[Fibre], box_min_um: Sequence[float],
                       side_um: float = 150.0, surface: str = "outer",
                       g_ratio: float = 0.5, min_inside_length_um: float = 20.0) -> dict:
    """Compute all JSON values. Returns a dict of config keys plus diagnostics."""
    seg = build_segments(fibres, box_min_um, side_um)
    ic = icvf(seg, surface, g_ratio)
    r_in = inner_radii(per_fibre_radius(seg, min_inside_length_um), surface, g_ratio)
    gam = fit_gamma(r_in)
    md = main_direction(seg)
    cfg = {
        "VoxelEdgeLength": side_um,
        "AxonsWithMyelinICVF": ic["AxonsWithMyelinICVF_percent"],
        "AxonsICVF": 0.0,
        "AlphaMyelin": gam["AlphaMyelin"],
        "BetaMyelin": gam["BetaMyelin"],
        "MinRadius": gam["min_observed_inner_radius_um"],
        "FODF_c2": c2_chord(seg, min_inside_length_um),
        **myelin_K_from_g_ratio(g_ratio),
    }
    diag = {"icvf": ic, "gamma": gam, "main_direction": md,
            "c2_segments": c2_segments(seg)}
    return {"config": cfg, "diagnostics": diag}


def _check_surface(surface: str) -> None:
    if surface not in ("inner", "outer"):
        raise ValueError("surface must be 'inner' or 'outer'")


def total_mesh_volume_um3(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Sum of signed volumes of the closed triangle meshes (divergence theorem).
    Cross-check for ICVF ONLY if all meshes are closed, consistently oriented and
    fully inside the voxel; otherwise it will not match the clipped ICVF."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(abs(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0))