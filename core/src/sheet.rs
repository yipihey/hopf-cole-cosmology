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
    for i in 0..n {
        for j in 0..n {
            let oi = if i + 1 == n { l } else { 0.0 };
            let oj = if j + 1 == n { l } else { 0.0 };
            let p00 = get(i, j, 0.0, 0.0);
            let p10 = get(i + 1, j, oi, 0.0);
            let p01 = get(i, j + 1, 0.0, oj);
            let p11 = get(i + 1, j + 1, oi, oj);
            for tri in [[p00, p10, p11], [p00, p11, p01]] {
                rasterize_tri(&tri, mass, ne, dxe, l, ss, wsub, &mut rho);
            }
        }
    }
    rho.iter().map(|&v| v as f32).collect()
}

#[allow(clippy::too_many_arguments)]
fn rasterize_tri(t: &[(f64, f64); 3], mass: f64, ne: usize, dxe: f64, l: f64, ss: usize, wsub: f64, rho: &mut [f64]) {
    let (x0, y0) = t[0];
    let (x1, y1) = t[1];
    let (x2, y2) = t[2];
    let area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if area2.abs() < 1e-300 {
        return;
    }
    let dens = mass / (0.5 * area2.abs());
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
            let mut cov = 0usize;
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
                        cov += 1;
                    }
                }
            }
            if cov > 0 {
                rho[iw * ne + jw] += dens * cov as f64 * wsub;
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
