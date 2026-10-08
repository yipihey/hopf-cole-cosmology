//! Eulerian density estimates from the displaced Lagrangian grid:
//! (a) phase-space-sheet rasterization (2D: triangles, point-sampled, exact
//!     multi-stream sum of 1/|J| per stream), and
//! (b) cloud-in-cell particle deposit (any dimension).

use crate::grid::Grid;

/// 2D sheet density: each Lagrangian cell is split into two triangles that
/// carry mass dq²/2 each.  At each Eulerian sample point we add m/|A| for
/// every triangle (stream) covering it.  `ss` = supersampling per axis.
/// `pos` is interleaved (x, y) in box units, unwrapped.  Output: ρ/ρ̄ on an
/// `ne × ne` Eulerian grid.
pub fn sheet_density_2d(grid: &Grid, pos: &[f32], ne: usize, ss: usize) -> Vec<f32> {
    sheet_density_2d_weighted(grid, pos, None, ne, ss)
}

/// Like `sheet_density_2d`, but with a density *shape* interpolated linearly
/// from vertex values `wv` (e.g. 1/|J| at the Lagrangian grid points): inside
/// each triangle ρ(x) = c · Σ_i λ_i w_i with barycentric λ and the constant c
/// chosen so that the triangle still deposits exactly its mass
/// (∫ linear = |A| · mean(w_i)  ⇒  c = m / (|A| · mean(w))).  Second-order
/// accurate where the plain sheet is first order; exactly mass conserving.
pub fn sheet_density_2d_weighted(grid: &Grid, pos: &[f32], wv: Option<&[f32]>, ne: usize, ss: usize) -> Vec<f32> {
    assert_eq!(grid.dim, 2);
    let n = grid.n;
    let l = grid.l;
    let dxe = l / ne as f64;
    let ss = ss.max(1);
    let mut rho = vec![0.0f64; ne * ne];
    let mass = (grid.dx() * grid.dx()) * 0.5; // mass of a triangle, ρ̄ = 1 units
    let wsub = 1.0 / (ss * ss) as f64;
    let get = |i: usize, j: usize, oi: f64, oj: f64| -> (f64, f64) {
        // vertex at Lagrangian (i,j) with periodic offsets added
        let idx = ((i % n) * n + (j % n)) * 2;
        (pos[idx] as f64 + oi, pos[idx + 1] as f64 + oj)
    };
    let wgt = |i: usize, j: usize| -> f64 { match wv { Some(w) => w[(i % n) * n + (j % n)] as f64, None => 1.0 } };
    for i in 0..n {
        for j in 0..n {
            let oi = if i + 1 == n { l } else { 0.0 };
            let oj = if j + 1 == n { l } else { 0.0 };
            let p00 = get(i, j, 0.0, 0.0);
            let p10 = get(i + 1, j, oi, 0.0);
            let p01 = get(i, j + 1, 0.0, oj);
            let p11 = get(i + 1, j + 1, oi, oj);
            let (w00, w10, w01, w11) = (wgt(i, j), wgt(i + 1, j), wgt(i, j + 1), wgt(i + 1, j + 1));
            for (tri, w) in [([p00, p10, p11], [w00, w10, w11]), ([p00, p11, p01], [w00, w11, w01])] {
                rasterize_tri(&tri, if wv.is_some() { Some(w) } else { None }, mass, ne, dxe, l, ss, wsub, &mut rho);
            }
        }
    }
    rho.iter().map(|&v| v as f32).collect()
}

#[allow(clippy::too_many_arguments)]
fn rasterize_tri(t: &[(f64, f64); 3], wv: Option<[f64; 3]>, mass: f64, ne: usize, dxe: f64, l: f64, ss: usize, wsub: f64, rho: &mut [f64]) {
    let (x0, y0) = t[0];
    let (x1, y1) = t[1];
    let (x2, y2) = t[2];
    let area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if area2.abs() < 1e-300 {
        return;
    }
    let dens = mass / (0.5 * area2.abs());
    // linear shape: ρ = dens · (λ·w) / mean(w)   (integrates to the triangle mass)
    let (w, wmean) = match wv { Some(w) => { let m = (w[0] + w[1] + w[2]) / 3.0; if m > 0.0 { (w, m) } else { ([1.0, 1.0, 1.0], 1.0) } } None => ([1.0, 1.0, 1.0], 1.0) };
    let xmin = x0.min(x1).min(x2);
    let xmax = x0.max(x1).max(x2);
    let ymin = y0.min(y1).min(y2);
    let ymax = y0.max(y1).max(y2);
    // sample points at (i + (s+0.5)/ss) * dxe
    let i0 = ((xmin / dxe) - 1.0).floor() as i64;
    let i1 = ((xmax / dxe) + 1.0).ceil() as i64;
    let j0 = ((ymin / dxe) - 1.0).floor() as i64;
    let j1 = ((ymax / dxe) + 1.0).ceil() as i64;
    let inv = 1.0 / area2;
    for i in i0..=i1 {
        let iw = i.rem_euclid(ne as i64) as usize;
        for j in j0..=j1 {
            let jw = j.rem_euclid(ne as i64) as usize;
            let mut acc = 0.0f64;
            for si in 0..ss {
                let px = (i as f64 + (si as f64 + 0.5) / ss as f64) * dxe;
                if px < xmin || px > xmax {
                    continue;
                }
                for sj in 0..ss {
                    let py = (j as f64 + (sj as f64 + 0.5) / ss as f64) * dxe;
                    if py < ymin || py > ymax {
                        continue;
                    }
                    // barycentric test
                    let w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv;
                    let w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv;
                    let w2 = 1.0 - w0 - w1;
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        acc += (w0 * w[0] + w1 * w[1] + w2 * w[2]) / wmean;
                    }
                }
            }
            if acc > 0.0 {
                rho[iw * ne + jw] += dens * acc * wsub;
            }
        }
    }
    let _ = l;
}

/// Cloud-in-cell deposit of unit-mass particles (interleaved positions) onto
/// an `ne^dim` grid.  Returns ρ/ρ̄.
pub fn cic_density(dim: usize, pos: &[f32], npart: usize, l: f64, ne: usize) -> Vec<f32> {
    let size = ne.pow(dim as u32);
    let mut rho = vec![0.0f64; size];
    let dxe = l / ne as f64;
    let mut i0 = [0usize; 3];
    let mut w1 = [0.0f64; 3];
    for p in 0..npart {
        for a in 0..dim {
            let x = pos[p * dim + a] as f64 / dxe - 0.5;
            let f = x.floor();
            w1[a] = x - f;
            i0[a] = (f as i64).rem_euclid(ne as i64) as usize;
        }
        let ncorner = 1usize << dim;
        for c in 0..ncorner {
            let mut w = 1.0;
            let mut idx = 0usize;
            for a in 0..dim {
                let bit = (c >> a) & 1;
                let ia = (i0[a] + bit) % ne;
                w *= if bit == 1 { w1[a] } else { 1.0 - w1[a] };
                idx = idx * ne + ia;
            }
            rho[idx] += w;
        }
    }
    let norm = size as f64 / npart as f64;
    rho.iter().map(|&v| (v * norm) as f32).collect()
}

/// CIC window deconvolution factor for the power spectrum (Jing 2005 leading term):
/// W(k) = Π_a sinc(k_a dx/2)^2.
pub fn cic_window(grid: &Grid, idx: usize) -> f64 {
    let ijk = grid.unravel(idx);
    let mut w = 1.0;
    for a in 0..grid.dim {
        let x = std::f64::consts::PI * grid.ifreq(ijk[a]) as f64 / grid.n as f64;
        let s = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
        w *= s * s;
    }
    w
}

/// 3D sheet density: each Lagrangian cube is split into 6 Kuhn tetrahedra of
/// mass dq³/6; every Eulerian sample point inside a tetrahedron receives
/// m/|V| (sum over streams).  `pos` interleaved (x,y,z), unwrapped.  Output
/// ρ/ρ̄ on an `ne³` grid; `ss` = supersampling per axis.
pub fn sheet_density_3d(grid: &Grid, pos: &[f32], ne: usize, ss: usize) -> Vec<f32> {
    sheet_density_3d_weighted(grid, pos, None, ne, ss)
}

/// 3D version of `sheet_density_2d_weighted` (linear shape from vertex values inside each tetrahedron).
pub fn sheet_density_3d_weighted(grid: &Grid, pos: &[f32], wv: Option<&[f32]>, ne: usize, ss: usize) -> Vec<f32> {
    assert_eq!(grid.dim, 3);
    let n = grid.n;
    let l = grid.l;
    let dxe = l / ne as f64;
    let ss = ss.max(1);
    let mut rho = vec![0.0f64; ne * ne * ne];
    let mass = grid.dx().powi(3) / 6.0;
    let wsub = 1.0 / (ss * ss * ss) as f64;
    let get = |i: usize, j: usize, k: usize, off: [f64; 3]| -> [f64; 3] {
        let idx = (((i % n) * n + (j % n)) * n + (k % n)) * 3;
        [pos[idx] as f64 + off[0], pos[idx + 1] as f64 + off[1], pos[idx + 2] as f64 + off[2]]
    };
    // Kuhn (Freudenthal) decomposition of the unit cube along the main diagonal
    const TETS: [[usize; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
    for i in 0..n {
        let oi = if i + 1 == n { l } else { 0.0 };
        for j in 0..n {
            let oj = if j + 1 == n { l } else { 0.0 };
            for k in 0..n {
                let ok = if k + 1 == n { l } else { 0.0 };
                // cube corners indexed by bits (x,y,z)
                let mut c = [[0.0f64; 3]; 8];
                let mut cw = [1.0f64; 8];
                for b in 0..8 {
                    let (bx, by, bz) = (b & 1, (b >> 1) & 1, (b >> 2) & 1);
                    c[b] = get(i + bx, j + by, k + bz, [if bx == 1 { oi } else { 0.0 }, if by == 1 { oj } else { 0.0 }, if bz == 1 { ok } else { 0.0 }]);
                    if let Some(w) = wv { cw[b] = w[(((i + bx) % n) * n + ((j + by) % n)) * n + ((k + bz) % n)] as f64; }
                }
                for t in TETS.iter() {
                    let w = if wv.is_some() { Some([cw[t[0]], cw[t[1]], cw[t[2]], cw[t[3]]]) } else { None };
                    rasterize_tet(&[c[t[0]], c[t[1]], c[t[2]], c[t[3]]], w, mass, ne, dxe, ss, wsub, &mut rho);
                }
            }
        }
    }
    rho.iter().map(|&v| v as f32).collect()
}

fn rasterize_tet(p: &[[f64; 3]; 4], wv: Option<[f64; 4]>, mass: f64, ne: usize, dxe: f64, ss: usize, wsub: f64, rho: &mut [f64]) {
    // signed volume ×6
    let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let e3 = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
    if det.abs() < 1e-300 { return; }
    let dens = mass / (det.abs() / 6.0);
    let (w, wmean) = match wv { Some(w) => { let m = (w[0] + w[1] + w[2] + w[3]) / 4.0; if m > 0.0 { (w, m) } else { ([1.0; 4], 1.0) } } None => ([1.0; 4], 1.0) };
    let inv = 1.0 / det;
    // inverse of the edge matrix (columns e1,e2,e3) for barycentric coordinates
    let m = [
        [(e2[1] * e3[2] - e2[2] * e3[1]) * inv, (e3[0] * e2[2] - e3[2] * e2[0]) * inv, (e2[0] * e3[1] - e2[1] * e3[0]) * inv],
        [(e3[1] * e1[2] - e3[2] * e1[1]) * inv, (e1[0] * e3[2] - e1[2] * e3[0]) * inv, (e3[0] * e1[1] - e3[1] * e1[0]) * inv],
        [(e1[1] * e2[2] - e1[2] * e2[1]) * inv, (e2[0] * e1[2] - e2[2] * e1[0]) * inv, (e1[0] * e2[1] - e1[1] * e2[0]) * inv],
    ];
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in p.iter() { for a in 0..3 { lo[a] = lo[a].min(v[a]); hi[a] = hi[a].max(v[a]); } }
    let c0: Vec<i64> = (0..3).map(|a| ((lo[a] / dxe) - 1.0).floor() as i64).collect();
    let c1: Vec<i64> = (0..3).map(|a| ((hi[a] / dxe) + 1.0).ceil() as i64).collect();
    let ne_i = ne as i64;
    for ci in c0[0]..=c1[0] {
        let iw = ci.rem_euclid(ne_i) as usize;
        for cj in c0[1]..=c1[1] {
            let jw = cj.rem_euclid(ne_i) as usize;
            for ck in c0[2]..=c1[2] {
                let kw = ck.rem_euclid(ne_i) as usize;
                let mut acc = 0.0f64;
                for si in 0..ss { let px = (ci as f64 + (si as f64 + 0.5) / ss as f64) * dxe; if px < lo[0] || px > hi[0] { continue; }
                    for sj in 0..ss { let py = (cj as f64 + (sj as f64 + 0.5) / ss as f64) * dxe; if py < lo[1] || py > hi[1] { continue; }
                        for sk in 0..ss { let pz = (ck as f64 + (sk as f64 + 0.5) / ss as f64) * dxe; if pz < lo[2] || pz > hi[2] { continue; }
                            let r = [px - p[0][0], py - p[0][1], pz - p[0][2]];
                            let b1 = m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2];
                            let b2 = m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2];
                            let b3 = m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2];
                            if b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0 && b1 + b2 + b3 <= 1.0 {
                                let b0 = 1.0 - b1 - b2 - b3;
                                acc += (b0 * w[0] + b1 * w[1] + b2 * w[2] + b3 * w[3]) / wmean;
                            }
                        }
                    }
                }
                if acc > 0.0 { rho[(iw * ne + jw) * ne + kw] += dens * acc * wsub; }
            }
        }
    }
}
