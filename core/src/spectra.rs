//! Power spectra, Fourier-space display maps, and standard / Zel'dovich
//! perturbation theory (1-loop) predictions.

use crate::fft::FftEngine;
use crate::grid::Grid;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

pub struct Spectrum {
    pub k: Vec<f64>,
    pub p: Vec<f64>,
    pub nmodes: Vec<f64>,
}

/// Bin edges: log-spaced from k_f to √dim k_nyq (nbins bins).
pub fn bin_edges(grid: &Grid, nbins: usize) -> Vec<f64> {
    let kmin = grid.kf() * 0.9;
    let kmax = grid.knyq() * (grid.dim as f64).sqrt() * 1.01;
    (0..=nbins).map(|i| kmin * (kmax / kmin).powf(i as f64 / nbins as f64)).collect()
}

/// Binned (cross-)power spectrum of two Fourier fields (forward unnormalized FFT),
/// P(k) = L^d Re[a b*] / N^{2d}.
pub fn power_spectrum_hat(grid: &Grid, a: &[C64], b: &[C64], nbins: usize, window: Option<&dyn Fn(usize) -> f64>) -> Spectrum {
    let edges = bin_edges(grid, nbins);
    let mut ksum = vec![0.0; nbins];
    let mut psum = vec![0.0; nbins];
    let mut cnt = vec![0.0; nbins];
    let norm = grid.l.powi(grid.dim as i32) / (grid.size as f64).powi(2);
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    for idx in 1..grid.size {
        let k = grid.k2(idx).sqrt();
        if k < edges[0] || k >= edges[nbins] {
            continue;
        }
        let bi = ((k.ln() - lk0) / dlk).floor() as usize;
        let bi = bi.min(nbins - 1);
        let mut v = (a[idx] * b[idx].conj()).re * norm;
        if let Some(w) = window {
            v /= w(idx);
        }
        ksum[bi] += k;
        psum[bi] += v;
        cnt[bi] += 1.0;
    }
    let mut k = Vec::new();
    let mut p = Vec::new();
    let mut nm = Vec::new();
    for i in 0..nbins {
        if cnt[i] > 0.0 {
            k.push(ksum[i] / cnt[i]);
            p.push(psum[i] / cnt[i]);
            nm.push(cnt[i]);
        }
    }
    Spectrum { k, p, nmodes: nm }
}

pub fn power_spectrum(grid: &Grid, eng: &mut FftEngine, f: &[f32], nbins: usize) -> Spectrum {
    let fh = eng.forward_real_f32(f);
    power_spectrum_hat(grid, &fh, &fh, nbins, None)
}

/// Fourier display maps of a real field: (log10 |f̂|/max, phase).  2D: the full
/// fft-shifted n×n plane; 3D: the k_z = 0 plane.  Row = k_x index, col = k_y.
pub fn fourier_maps(grid: &Grid, fhat: &[C64]) -> (Vec<f32>, Vec<f32>) {
    let n = grid.n;
    let mut amp = vec![0.0f32; n * n];
    let mut ph = vec![0.0f32; n * n];
    let mut maxabs = 0.0f64;
    for idx in 0..grid.size {
        maxabs = maxabs.max(fhat[idx].norm());
    }
    let maxabs = maxabs.max(1e-300);
    for i in 0..n {
        for j in 0..n {
            // shifted indices: display (i,j) corresponds to freq i-n/2, j-n/2
            let fi = (i + n / 2) % n;
            let fj = (j + n / 2) % n;
            let idx = match grid.dim {
                2 => grid.ravel([fi, fj, 0]),
                3 => grid.ravel([fi, fj, 0]),
                _ => fi,
            };
            let v = fhat[idx];
            amp[i * n + j] = ((v.norm() / maxabs).max(1e-30)).log10() as f32;
            ph[i * n + j] = v.arg() as f32;
        }
    }
    (amp, ph)
}

// ---------------------------------------------------------------------------
// Perturbation theory kernels
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn add(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Unsymmetrized SPT kernels (F_n, G_n) by the Goroff et al. recursion (EdS).
pub fn fg_kernel(ks: &[[f64; 3]]) -> (f64, f64) {
    let n = ks.len();
    if n == 1 {
        return (1.0, 1.0);
    }
    let nf = n as f64;
    let pref = 1.0 / ((2.0 * nf + 3.0) * (nf - 1.0));
    let mut f = 0.0;
    let mut g = 0.0;
    for m in 1..n {
        let (_, g1) = fg_kernel(&ks[..m]);
        let (f2, g2) = fg_kernel(&ks[m..]);
        let mut k1 = [0.0; 3];
        for k in &ks[..m] {
            k1 = add(&k1, k);
        }
        let mut k2 = [0.0; 3];
        for k in &ks[m..] {
            k2 = add(&k2, k);
        }
        let k1s = dot(&k1, &k1);
        let k2s = dot(&k2, &k2);
        if k1s < 1e-300 || k2s < 1e-300 {
            continue;
        }
        let k12 = add(&k1, &k2);
        let alpha = dot(&k12, &k1) / k1s;
        let beta = dot(&k12, &k12) * dot(&k1, &k2) / (2.0 * k1s * k2s);
        f += g1 * ((2.0 * nf + 1.0) * alpha * f2 + 2.0 * beta * g2);
        g += g1 * (3.0 * alpha * f2 + 2.0 * nf * beta * g2);
    }
    (pref * f, pref * g)
}

/// Symmetrized F_n (average over all permutations; n ≤ 3 used here).
pub fn f_sym(ks: &[[f64; 3]]) -> f64 {
    let n = ks.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut total = 0.0;
    let mut count = 0.0;
    // Heap's algorithm
    fn heap(k: usize, idx: &mut Vec<usize>, ks: &[[f64; 3]], total: &mut f64, count: &mut f64) {
        if k == 1 {
            let perm: Vec<[f64; 3]> = idx.iter().map(|&i| ks[i]).collect();
            *total += fg_kernel(&perm).0;
            *count += 1.0;
            return;
        }
        heap(k - 1, idx, ks, total, count);
        for i in 0..k - 1 {
            if k % 2 == 0 {
                idx.swap(i, k - 1);
            } else {
                idx.swap(0, k - 1);
            }
            heap(k - 1, idx, ks, total, count);
        }
    }
    heap(n, &mut idx, ks, &mut total, &mut count);
    total / count
}

/// Zel'dovich kernels F_n^ZA = (1/n!) Π (k·k_i / k_i²).
pub fn f_za(ks: &[[f64; 3]]) -> f64 {
    let mut k = [0.0; 3];
    for v in ks {
        k = add(&k, v);
    }
    let mut p = 1.0;
    let mut fact = 1.0;
    for (i, v) in ks.iter().enumerate() {
        let vs = dot(v, v);
        if vs < 1e-300 {
            return 0.0;
        }
        p *= dot(&k, v) / vs;
        fact *= (i + 1) as f64;
    }
    p / fact
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kernels {
    Spt,
    Zeldovich,
}

/// One-loop corrections (P22, P13) at wavenumber k for a linear P(k), in
/// dim = 2 (fields invariant along z) or 3.  Quadrature: log-spaced |q|,
/// uniform angles.
pub fn one_loop(pk: &dyn Fn(f64) -> f64, dim: usize, k: f64, kernels: Kernels, qmin: f64, qmax: f64, nq: usize, nth: usize) -> (f64, f64) {
    let kvec = [k, 0.0, 0.0];
    let mut p22 = 0.0;
    let mut p13 = 0.0;
    let dlq = (qmax / qmin).ln() / nq as f64;
    for iq in 0..nq {
        let q = qmin * ((iq as f64 + 0.5) * dlq).exp();
        let pq = pk(q);
        if pq == 0.0 {
            continue;
        }
        // measure: 2D: q dq dθ/(2π)² ; 3D: q² dq dμ dφ/(2π)³ → 2π q² dq dμ/(2π)³ (azimuthal symmetry)
        let (nang, wmeas) = if dim == 2 { (nth, q * q * dlq / (2.0 * PI).powi(2)) } else { (nth, q * q * q * dlq * 2.0 * PI / (2.0 * PI).powi(3)) };
        for ia in 0..nang {
            let (qv, wang) = if dim == 2 {
                let th = 2.0 * PI * (ia as f64 + 0.5) / nang as f64;
                ([q * th.cos(), q * th.sin(), 0.0], 2.0 * PI / nang as f64)
            } else {
                let mu = -1.0 + 2.0 * (ia as f64 + 0.5) / nang as f64;
                let s = (1.0 - mu * mu).max(0.0).sqrt();
                ([q * mu, q * s, 0.0], 2.0 / nang as f64)
            };
            let w = wmeas * wang;
            let kmq = [kvec[0] - qv[0], kvec[1] - qv[1], kvec[2] - qv[2]];
            let kmq_abs = dot(&kmq, &kmq).sqrt();
            let pkmq = pk(kmq_abs);
            let mq = [-qv[0], -qv[1], -qv[2]];
            match kernels {
                Kernels::Spt => {
                    if pkmq > 0.0 && kmq_abs > 1e-12 {
                        let f2 = f_sym(&[qv, kmq]);
                        p22 += 2.0 * f2 * f2 * pq * pkmq * w;
                    }
                    let f3 = f_sym(&[kvec, qv, mq]);
                    p13 += 6.0 * f3 * pq * w;
                }
                Kernels::Zeldovich => {
                    if pkmq > 0.0 && kmq_abs > 1e-12 {
                        let f2 = f_za(&[qv, kmq]);
                        p22 += 2.0 * f2 * f2 * pq * pkmq * w;
                    }
                    let f3 = f_za(&[kvec, qv, mq]);
                    p13 += 6.0 * f3 * pq * w;
                }
            }
        }
    }
    (p22, p13 * pk(k))
}

// ---------------------------------------------------------------------------
// Top-hat smoothing (one-point statistics)

/// Bessel J1 (Numerical Recipes rational approximation, |err| < 1e-7).
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let ans = (0.636619772 / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 { -ans } else { ans }
    }
}

/// Fourier-space window of a top-hat of radius r: disc (2D) or sphere (3D); 1D: box.
pub fn tophat_window(dim: usize, kr: f64) -> f64 {
    if kr < 1e-6 { return 1.0; }
    match dim {
        1 => kr.sin() / kr,
        2 => 2.0 * bessel_j1(kr) / kr,
        _ => 3.0 * (kr.sin() - kr * kr.cos()) / (kr * kr * kr),
    }
}

/// Smooth a real field with a top-hat of radius `radius` (box units) via the FFT.
pub fn tophat_smooth(grid: &Grid, eng: &mut FftEngine, f: &[f32], radius: f64) -> Vec<f32> {
    let mut h = eng.forward_real_f32(f);
    for idx in 0..grid.size {
        let k = grid.k2(idx).sqrt();
        h[idx] *= tophat_window(grid.dim, k * radius);
    }
    eng.inverse_to_real_f32(h)
}

// ---------------------------------------------------------------------------
// Viscous (adhesion-model) second-order kernel

/// Second-order Eulerian density kernel of the viscous Burgers / adhesion
/// dynamics (density defined through the inverse map, 1+δ = det(I - D∇∇Φ)):
///
///   δ̂₂(k) = D² ∫ F₂^ν(k₁,k₂; ν, D) δ̂₀(k₁) δ̂₀(k₂),
///   F₂^ν = [ ½ k² (k₁·k₂) W₂ + ½ (k₁²k₂² − (k₁·k₂)²) e^{-ν(k₁²+k₂²)D} ] / (k₁² k₂²),
///   W₂   = [e^{-ν(k₁²+k₂²)D} − e^{-νk²D}] / (2ν (k₁·k₂) D)   (→ 1 as ν → 0),
///
/// which reduces to the Zel'dovich kernel F₂^ZA = ½ (k·k₁/k₁²)(k·k₂/k₂²) for ν → 0.
/// The first-order propagator is e^{-νk²D}.
pub fn f2_viscous(k1: [f64; 3], k2: [f64; 3], nu: f64, d: f64) -> f64 {
    let k = add(&k1, &k2);
    let k1s = dot(&k1, &k1);
    let k2s = dot(&k2, &k2);
    let ks = dot(&k, &k);
    let k12 = dot(&k1, &k2);
    if k1s < 1e-300 || k2s < 1e-300 { return 0.0; }
    let e12 = (-nu * (k1s + k2s) * d).exp();
    let w2 = if (nu * k12 * d).abs() < 1e-12 {
        // limit: (e^{-νs D} − e^{-νk²D})/(2νk₁·k₂D) with k² − s = 2 k₁·k₂  →  e^{-νsD}·(1 - ...) ≈ e^{-ν s D}
        e12
    } else {
        (e12 - (-nu * ks * d).exp()) / (2.0 * nu * k12 * d)
    };
    (0.5 * ks * k12 * w2 + 0.5 * (k1s * k2s - k12 * k12) * e12) / (k1s * k2s)
}

/// Gravitational F₂ (EdS), Zel'dovich F₂^ZA, or viscous F₂^ν: kind 0, 1, 2.
pub fn f2_kernel(kind: u32, k1: [f64; 3], k2: [f64; 3], nu: f64, d: f64) -> f64 {
    match kind {
        0 => f_sym(&[k1, k2]),
        1 => f_za(&[k1, k2]),
        _ => f2_viscous(k1, k2, nu, d),
    }
}
