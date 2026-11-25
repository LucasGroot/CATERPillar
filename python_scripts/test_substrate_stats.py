"""Validate substrate_stats.py on synthetic tube meshes with known ground truth."""
import sys, tempfile, os
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substrate_stats as ss

rng = np.random.default_rng(1)
SIDE, G, NT = 150.0, 0.5, 32
ALPHA, BETA = 4.0, 0.25            # true inner-radius Gamma (um)
N_FIB = 300
MIN_R = 0.15


def make_fibre(r_out, direction, start):
    """Straight tube, ring every ~5 um, long enough to cross the voxel."""
    d = direction / np.linalg.norm(direction)
    n_ring = 60
    s = np.linspace(0, 300, n_ring)
    centres = start[None, :] + s[:, None] * d[None, :]
    # orthonormal frame
    a = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    th = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    ring = r_out * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    verts = (centres[:, None, :] + ring[None, :, :]).reshape(-1, 3)
    faces = []
    for i in range(n_ring - 1):
        for j in range(NT):
            a0, a1 = i * NT + j, i * NT + (j + 1) % NT
            b0, b1 = (i + 1) * NT + j, (i + 1) * NT + (j + 1) % NT
            faces += [(a0, a1, b1), (a0, b1, b0)]
    # end caps: two extra trailing vertices (cap centres)
    c0, c1 = len(verts), len(verts) + 1
    verts = np.vstack([verts, centres[0], centres[-1]])
    for j in range(NT):
        faces.append((c0, (j + 1) % NT, j))
        k = (n_ring - 1) * NT
        faces.append((c1, k + j, k + (j + 1) % NT))
    return verts, np.array(faces), centres, d


r_in = rng.gamma(ALPHA, BETA, N_FIB * 2)
r_in = r_in[r_in > MIN_R][:N_FIB]
r_out = r_in / G

all_v, all_f, truth = [], [], []
off = 0
for k in range(N_FIB):
    tilt = rng.normal(0, 0.15)          # rad
    az = rng.uniform(0, 2 * np.pi)
    d = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    xy = rng.uniform(0, SIDE, 2)
    # start below the voxel so every fibre crosses it (box is [0,150]^3)
    start = np.array([xy[0], xy[1], 0.0]) - 80 * d
    v, f, c, dd = make_fibre(r_out[k], d, start)
    all_v.append(v); all_f.append(f + off); off += len(v)
    truth.append((c, dd, r_out[k]))
V = np.vstack(all_v); F = np.vstack(all_f)

# write in mm, typical for meshing tools
tmp = os.path.join(tempfile.mkdtemp(), "synthetic.ply")
vert = np.array([tuple(p) for p in V * 1e-3], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
face = np.array([(tuple(t),) for t in F], dtype=[("vertex_indices", "i4", (3,))])
PlyData([PlyElement.describe(vert, "vertex"), PlyElement.describe(face, "face")]).write(tmp)

print("--- inspect_ply ---")
ss.inspect_ply(tmp, n_theta=NT)

fibres = ss.fibres_from_ply(tmp, n_theta=NT, scale_to_um=1000.0)
res = ss.caterpillar_values(fibres, box_min_um=(0, 0, 0), side_um=SIDE, surface="outer", g_ratio=G)

# ---------- ground truth ----------
# ICVF reference: Monte-Carlo point-in-cylinder, summed per fibre (no overlap correction)
P = rng.uniform(0, SIDE, (400000, 3))
vol_ref = 0.0
for c, d, ro in truth:
    rel = P - c[0]
    along = rel @ d
    perp = np.linalg.norm(rel - along[:, None] * d, axis=1)
    inside = (perp < ro) & (along >= 0) & (along <= 300)
    vol_ref += inside.mean() * SIDE ** 3
icvf_ref = vol_ref / SIDE ** 3
# c2 reference (chord = direction for straight fibres)
c2_ref = np.mean([d[2] ** 2 for _, d, _ in truth])

cfg, diag = res["config"], res["diagnostics"]
print("\n--- recovered vs truth ---")
print(f"ICVF outer   : {diag['icvf']['icvf_outer']:.4f}   truth (MC) {icvf_ref:.4f}")
print(f"Gamma alpha  : {cfg['AlphaMyelin']:.3f}   truth {ALPHA}")
print(f"Gamma beta   : {cfg['BetaMyelin']:.3f}   truth {BETA}")
print(f"mean r_inner : {diag['gamma']['observed_mean_um']:.3f}   truth sample mean {r_in.mean():.3f}")
print(f"c2 (chord)   : {cfg['FODF_c2']:.4f}   truth {c2_ref:.4f}")
print(f"c2 segments  : {diag['c2_segments']:.4f}")
print(f"main dir angle to z: {diag['main_direction']['angle_to_z_deg']:.2f} deg")
print("K1,K2,K3     :", cfg["K1"], round(cfg["K2"], 4), cfg["K3"])
print("fibres in voxel:", diag["icvf"]["n_fibres_in_voxel"], "of", N_FIB)

assert abs(diag["icvf"]["icvf_outer"] - icvf_ref) / icvf_ref < 0.03
assert abs(cfg["FODF_c2"] - c2_ref) < 0.002
assert abs(diag["gamma"]["observed_mean_um"] - r_in.mean()) / r_in.mean() < 0.02
print("\nAll checks passed.")