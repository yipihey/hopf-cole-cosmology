//! Deposit-free spectra of the phase-space sheet.
//!
//! 1. `direct_spectrum`: the exact Fourier transform of the piecewise-constant
//!    (P0) or piecewise-linear (P1) sheet density, via the Hermite–Genocchi
//!    identity  ∫_T e^{-ik·x} d^dx = d! |T| · exp[s_0, …, s_d],  s_j = -i k·v_j,
//!    where exp[…] is the divided difference of the exponential at the vertex
//!    phases (one extra repeated node for a P1 shape function).  Evaluated on
//!    sampled lattice modes per |k| bin: no Eulerian grid, no window, no aliasing.
//! 2. `nufft_density`: the Lagrangian integral ∫ d^dq e^{-ik·x(q)} of the
//!    spectrally refined (Fourier-interpolated) map, evaluated with the periodic
//!    trapezoidal rule on the fine grid through a type-1 NUFFT (Gaussian
//!    spreading on a 2× oversampled grid, FFT, deconvolution).  Exponentially
//!    accurate for a band-limited displacement; counts all streams.

use crate::fft::FftEngine;
use crate::grid::Grid;
use crate::rng::Rng;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Divided difference of exp at complex nodes `s` (stable for coincident nodes).
pub fn exp_divided_difference(s: &[C64]) -> C64 {
    let n = s.len();
    if n == 1 { return s[0].exp(); }
    // minimum pairwise separation
    let mut dmin = f64::INFINITY;
    for i in 0..n { for j in (i + 1)..n { dmin = dmin.min((s[i] - s[j]).norm()); } }
    if dmin > 0.05 {
        // Newton recursion on a table
        let mut t: Vec<C64> = s.iter().map(|v| v.exp()).collect();
        for level in 1..n {
            for i in 0..(n - level) {
                t[i] = (t[i + 1] - t[i]) / (s[i + level] - s[i]);
            }
        }
        t[0]
    } else {
        // Taylor expansion about the mean node: exp[s_0..s_{n-1}] = e^{s̄} Σ_m h_m(δ)/(m+n-1)!
        // with h_m the complete homogeneous symmetric polynomials of δ_j = s_j - s̄.
        let mut mean = C64::new(0.0, 0.0);
        for v in s { mean += v; }
        mean /= n as f64;
        let d: Vec<C64> = s.iter().map(|v| v - mean).collect();
        // h_m via the recursion over variables: h_m(x_1..x_k) = h_m(x_1..x_{k-1}) + x_k h_{m-1}(x_1..x_k)
        let mmax = 24;
        let mut h = vec![C64::new(0.0, 0.0); mmax + 1]; // h for the first k variables
        h[0] = C64::new(1.0, 0.0);
        for k in 0..n {
            for m in 1..=mmax { h[m] = h[m] + d[k] * h[m - 1]; }
        }
        let mut fact = 1.0f64;
        for i in 1..n { fact *= i as f64; } // (n-1)!
        let mut sum = C64::new(0.0, 0.0);
        let mut f = fact;
        for m in 0..=mmax {
            if m > 0 { f *= (m + n - 1) as f64; }
            let term = h[m] / f;
            sum += term;
            if term.norm() < 1e-18 * sum.norm().max(1e-300) && m > 2 { break; }
        }
        mean.exp() * sum
    }
}

/// Fourier transform ∫_T w(x) e^{-ik·x} d^dx of a simplex with vertices `v` and a
/// linear density w interpolated from vertex values `wv` (all equal 1 for P0).
pub fn simplex_transform(v: &[[f64; 3]], wv: &[f64], dim: usize, k: [f64; 3]) -> C64 {
    let n = v.len(); // dim + 1
    let mut s = Vec::with_capacity(n + 1);
    for p in v { s.push(C64::new(0.0, -(k[0] * p[0] + k[1] * p[1] + k[2] * p[2]))); }
    let vol = simplex_volume(v, dim);
    let fact = match dim { 2 => 2.0, 3 => 6.0, _ => 1.0 };
    let mut total = C64::new(0.0, 0.0);
    let p0 = wv.iter().all(|&w| (w - wv[0]).abs() < 1e-15);
    if p0 {
        total = exp_divided_difference(&s) * (fact * vol * wv[0]);
    } else {
        // Σ_j w_j ∫_T λ_j e^{-ik·x} = Σ_j w_j d! |T| exp[s_0..s_d, s_j]
        for j in 0..n {
            s.push(s[j]);
            total += exp_divided_difference(&s) * (fact * vol * wv[j]);
            s.pop();
        }
    }
    total
}

fn simplex_volume(v: &[[f64; 3]], dim: usize) -> f64 {
    if dim == 2 {
        0.5 * ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1])).abs()
    } else {
        let e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
        let e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
        let e3 = [v[3][0] - v[0][0], v[3][1] - v[0][1], v[3][2] - v[0][2]];
        (e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])).abs() / 6.0
    }
}

/// Lattice modes sampled per log bin: returns (bin edges, list of (bin, kvec, |k|)).
pub fn sample_modes(grid: &Grid, nbins: usize, per_bin: usize, seed: u64) -> (Vec<f64>, Vec<(usize, [f64; 3], f64)>) {
    let edges = crate::spectra::bin_edges(grid, nbins);
    let kf = grid.kf();
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    let n = grid.n as i64;
    let d = grid.dim;
    for b in 0..nbins {
        let (klo, khi) = (edges[b], edges[b + 1]);
        // enumerate when the shell is small, otherwise sample integer vectors in the bounding cube
        let mmax = (khi / kf).ceil() as i64;
        let count_est = if d == 2 { PI * (khi * khi - klo * klo) / (kf * kf) } else { 4.0 / 3.0 * PI * (khi.powi(3) - klo.powi(3)) / kf.powi(3) };
        if count_est <= per_bin as f64 * 1.5 {
            let r = mmax.min(n / 2);
            for mx in -r..=r { for my in -r..=r {
                let mzs: Vec<i64> = if d == 3 { (-r..=r).collect() } else { vec![0] };
                for mz in mzs {
                    let k = [mx as f64 * kf, my as f64 * kf, mz as f64 * kf];
                    let kk = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
                    if kk >= klo && kk < khi && kk > 0.0 { out.push((b, k, kk)); }
                }
            } }
        } else {
            let mut got = 0;
            let mut tries = 0;
            while got < per_bin && tries < per_bin * 200 {
                tries += 1;
                let mx = (rng.uniform() * (2.0 * mmax as f64 + 1.0)).floor() as i64 - mmax;
                let my = (rng.uniform() * (2.0 * mmax as f64 + 1.0)).floor() as i64 - mmax;
                let mz = if d == 3 { (rng.uniform() * (2.0 * mmax as f64 + 1.0)).floor() as i64 - mmax } else { 0 };
                if mx.abs() > n / 2 || my.abs() > n / 2 || mz.abs() > n / 2 { continue; }
                let k = [mx as f64 * kf, my as f64 * kf, mz as f64 * kf];
                let kk = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
                if kk >= klo && kk < khi && kk > 0.0 { out.push((b, k, kk)); got += 1; }
            }
        }
    }
    (edges, out)
}

/// Direct sheet spectrum: P(k) = |δ̂(k)|² / L^d with δ̂ summed over all simplices
/// (positions interleaved and unwrapped; `wv` = vertex densities for P1 or None).
/// Returns (k_eff, P, n_modes) per bin.
pub fn direct_spectrum(grid: &Grid, pos: &[f32], wv: Option<&[f32]>, nbins: usize, per_bin: usize, seed: u64) -> crate::spectra::Spectrum {
    let (_, modes) = sample_modes(grid, nbins, per_bin, seed);
    let d = grid.dim;
    let n = grid.n;
    let l = grid.l;
    let nm = modes.len();
    let mut acc = vec![C64::new(0.0, 0.0); nm];
    let simplex_mass = if d == 2 { grid.dx() * grid.dx() * 0.5 } else { grid.dx().powi(3) / 6.0 };
    let get = |ijk: [usize; 3], off: [f64; 3]| -> [f64; 3] {
        let mut idx = [0usize; 3];
        let mut shift = [0.0; 3];
        for a in 0..d { if ijk[a] >= n { idx[a] = ijk[a] - n; shift[a] = l; } else { idx[a] = ijk[a]; } }
        let flat = grid.ravel(idx);
        let mut p = [0.0; 3];
        for a in 0..d { p[a] = pos[flat * d + a] as f64 + shift[a] + off[a]; }
        p
    };
    let wgt = |ijk: [usize; 3]| -> f64 {
        match wv { Some(w) => { let mut idx = [0usize; 3]; for a in 0..d { idx[a] = ijk[a] % n; } w[grid.ravel(idx)] as f64 } None => 1.0 }
    };
    let mut process = |verts: &[[f64; 3]], w: &[f64]| {
        let vol = simplex_volume(verts, d);
        if vol < 1e-300 { return; }
        let dens = simplex_mass / vol;
        let wmean = w.iter().sum::<f64>() / w.len() as f64;
        let wn: Vec<f64> = if wv.is_some() { w.iter().map(|x| x / wmean).collect() } else { vec![1.0; w.len()] };
        for (mi, (_, k, _)) in modes.iter().enumerate() {
            acc[mi] += simplex_transform(verts, &wn, d, *k) * dens;
        }
    };
    if d == 2 {
        for i in 0..n { for j in 0..n {
            let p00 = get([i, j, 0], [0.0; 3]); let p10 = get([i + 1, j, 0], [0.0; 3]); let p01 = get([i, j + 1, 0], [0.0; 3]); let p11 = get([i + 1, j + 1, 0], [0.0; 3]);
            let (w00, w10, w01, w11) = (wgt([i, j, 0]), wgt([i + 1, j, 0]), wgt([i, j + 1, 0]), wgt([i + 1, j + 1, 0]));
            process(&[p00, p10, p11], &[w00, w10, w11]);
            process(&[p00, p11, p01], &[w00, w11, w01]);
        } }
    } else {
        const TETS: [[usize; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
        for i in 0..n { for j in 0..n { for kk in 0..n {
            let mut c = [[0.0; 3]; 8]; let mut cw = [1.0; 8];
            for b in 0..8 { let ijk = [i + (b & 1), j + ((b >> 1) & 1), kk + ((b >> 2) & 1)]; c[b] = get(ijk, [0.0; 3]); cw[b] = wgt(ijk); }
            for t in TETS.iter() { process(&[c[t[0]], c[t[1]], c[t[2]], c[t[3]]], &[cw[t[0]], cw[t[1]], cw[t[2]], cw[t[3]]]); }
        } } }
    }
    // bin
    let mut ksum = vec![0.0; nbins]; let mut psum = vec![0.0; nbins]; let mut cnt = vec![0.0; nbins];
    let norm = 1.0 / l.powi(d as i32);
    for (mi, (b, _, kk)) in modes.iter().enumerate() {
        ksum[*b] += kk; psum[*b] += acc[mi].norm_sqr() * norm; cnt[*b] += 1.0;
    }
    let (mut k, mut p, mut nmv) = (Vec::new(), Vec::new(), Vec::new());
    for b in 0..nbins { if cnt[b] > 0.0 { k.push(ksum[b] / cnt[b]); p.push(psum[b] / cnt[b]); nmv.push(cnt[b]); } }
    crate::spectra::Spectrum { k, p, nmodes: nmv }
}

/// Type-1 NUFFT density of the refined map: unit-mass points at the fine-grid
/// positions `pos` (interleaved, `npts` points) are spread with a Gaussian of
/// width `sigma` (in oversampled cells) onto a 2× oversampled grid, transformed,
/// deconvolved, truncated to the `ne` grid and transformed back.  Returns ρ/ρ̄.
pub fn nufft_density(dim: usize, pos: &[f32], npts: usize, l: f64, ne: usize, sigma: f64, support: usize) -> Vec<f32> {
    let no = 2 * ne;
    let ogrid = Grid::new(dim, no, l);
    let dxo = l / no as f64;
    let size = ogrid.size;
    let mut rho = vec![0.0f64; size];
    let w = support as i64;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut i0 = [0i64; 3];
    let mut fr = [0.0f64; 3];
    for p in 0..npts {
        for a in 0..dim { let x = pos[p * dim + a] as f64 / dxo; let f = x.floor(); i0[a] = f as i64; fr[a] = x - f; }
        // separable Gaussian weights
        let mut wts = [[0.0f64; 16]; 3];
        for a in 0..dim { for s in -w..=w { let dxs = s as f64 - fr[a] + 0.5 - 0.5; // offset to cell index s relative to i0 (cell centres at i+0.5?) use node convention
            let dd = dxs; wts[a][(s + w) as usize] = (-dd * dd * inv2s2).exp(); } }
        if dim == 2 {
            for si in -w..=w { let ii = (i0[0] + si).rem_euclid(no as i64) as usize; let wi = wts[0][(si + w) as usize];
                for sj in -w..=w { let jj = (i0[1] + sj).rem_euclid(no as i64) as usize; rho[ii * no + jj] += wi * wts[1][(sj + w) as usize]; } }
        } else {
            for si in -w..=w { let ii = (i0[0] + si).rem_euclid(no as i64) as usize; let wi = wts[0][(si + w) as usize];
                for sj in -w..=w { let jj = (i0[1] + sj).rem_euclid(no as i64) as usize; let wij = wi * wts[1][(sj + w) as usize];
                    for sk in -w..=w { let kk = (i0[2] + sk).rem_euclid(no as i64) as usize; rho[(ii * no + jj) * no + kk] += wij * wts[2][(sk + w) as usize]; } } }
        }
    }
    // normalize to mean 1 (unit masses, Gaussian weights sum to ~ (σ√2π)^d)
    let mean: f64 = rho.iter().sum::<f64>() / size as f64;
    for v in rho.iter_mut() { *v /= mean; }
    // FFT, deconvolve by the Gaussian kernel's transform exp(-k²σ²dxo²/2), truncate to the ne grid
    let mut eng = FftEngine::new(&ogrid);
    let rh = eng.forward_real(&rho);
    let fine = Grid::new(dim, ne, l);
    let mut eng_f = FftEngine::new(&fine);
    let mut out = vec![C64::new(0.0, 0.0); fine.size];
    for idx in 0..fine.size {
        let ijk = fine.unravel(idx);
        let mut oijk = [0usize; 3];
        let mut skip = false;
        for a in 0..dim {
            let f = fine.ifreq(ijk[a]);
            if ne % 2 == 0 && ijk[a] == ne / 2 { skip = true; }
            oijk[a] = if f >= 0 { f as usize } else { (no as i64 + f) as usize };
        }
        if skip { continue; }
        let k2 = fine.k2(idx);
        let win = (-0.5 * k2 * (sigma * dxo).powi(2)).exp();
        out[idx] = rh[ogrid.ravel(oijk)] / win * (fine.size as f64 / size as f64);
    }
    eng_f.inverse_to_real_f32(out)
}
