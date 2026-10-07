//! Initial conditions: linear power spectra and Gaussian random fields,
//! plus a few deterministic "textbook" configurations.

use crate::fft::FftEngine;
use crate::grid::Grid;
use crate::rng::Rng;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Shape of the linear power spectrum (before smoothing / normalization).
#[derive(Clone, Debug)]
pub enum PkShape {
    /// P(k) ∝ k^n
    PowerLaw { n: f64 },
    /// P(k) ∝ k^ns T_BBKS(k/Γ)^2 — CDM-like shape, Γ in units of (box units)^-1.
    Bbks { ns: f64, gamma: f64 },
    /// Eisenstein & Hu (1998) zero-baryon no-wiggle transfer function with
    /// an *effective* shape parameter Γ (in box units): k_eff = k/Γ.
    EhNoWiggle { ns: f64, gamma: f64 },
}

impl PkShape {
    /// Unnormalized P(k) shape.
    pub fn eval(&self, k: f64) -> f64 {
        if k <= 0.0 {
            return 0.0;
        }
        match *self {
            PkShape::PowerLaw { n } => k.powf(n),
            PkShape::Bbks { ns, gamma } => {
                let q = k / gamma;
                let t = (1.0 + 2.34 * q).ln() / (2.34 * q)
                    * (1.0 + 3.89 * q + (16.1 * q).powi(2) + (5.46 * q).powi(3) + (6.71 * q).powi(4)).powf(-0.25);
                k.powf(ns) * t * t
            }
            PkShape::EhNoWiggle { ns, gamma } => {
                // EH98 eq. 29-31 with the baryon-free shape; q = k/Γ in units
                // where Θ_2.7 factors are absorbed into Γ.
                let q = k / gamma;
                let l0 = (2.0 * std::f64::consts::E + 1.8 * q).ln();
                let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
                let t = l0 / (l0 + c0 * q * q);
                k.powf(ns) * t * t
            }
        }
    }
}

/// Full linear power spectrum model: shape × smoothing × amplitude.
#[derive(Clone, Debug)]
pub struct LinearPk {
    pub shape: PkShape,
    /// Gaussian smoothing scale R: P(k) → P(k) exp(-k^2 R^2).  R = 0: none.
    pub r_smooth: f64,
    /// Overall amplitude multiplying the shape.
    pub amp: f64,
    /// Optional sharp cutoff k_max (0 = none).
    pub k_cut: f64,
}

impl LinearPk {
    pub fn eval(&self, k: f64) -> f64 {
        if k <= 0.0 || (self.k_cut > 0.0 && k > self.k_cut) {
            return 0.0;
        }
        let w = if self.r_smooth > 0.0 { (-(k * self.r_smooth).powi(2)).exp() } else { 1.0 };
        self.amp * self.shape.eval(k) * w
    }
    /// Shape × smoothing only (amp = 1).
    pub fn eval_unnormalized(&self, k: f64) -> f64 {
        let w = if self.r_smooth > 0.0 { (-(k * self.r_smooth).powi(2)).exp() } else { 1.0 };
        if k <= 0.0 || (self.k_cut > 0.0 && k > self.k_cut) { 0.0 } else { self.shape.eval(k) * w }
    }
}

/// Variance σ^2 of the field for a given P(k) on the discrete grid:
/// σ^2 = (1/L^d) Σ_k P(k).
pub fn grid_variance(grid: &Grid, pk: &dyn Fn(f64) -> f64) -> f64 {
    let mut s = 0.0;
    for idx in 0..grid.size {
        let k = grid.k2(idx).sqrt();
        s += pk(k);
    }
    s / grid.l.powi(grid.dim as i32)
}

/// Gaussian random field δ0 in Fourier space with the given P(k).
/// Method: real white noise on the grid → FFT → × sqrt(P(k) N^d / L^d).
/// This is mode-stable w.r.t. the seed and exactly Hermitian.
pub fn gaussian_field_hat(grid: &Grid, eng: &mut FftEngine, pk: &LinearPk, seed: u64) -> Vec<C64> {
    let mut rng = Rng::new(seed);
    let white: Vec<f64> = (0..grid.size).map(|_| rng.normal()).collect();
    let mut hat = eng.forward_real(&white);
    let fac = (grid.size as f64 / grid.l.powi(grid.dim as i32)).sqrt();
    for idx in 0..grid.size {
        let k = grid.k2(idx).sqrt();
        let p = pk.eval(k);
        hat[idx] *= fac * p.sqrt();
    }
    hat[0] = C64::new(0.0, 0.0);
    hat
}

/// Deterministic configurations.
#[derive(Clone, Debug)]
pub enum Preset {
    /// δ0 = Σ_i A_i cos(k_i · x + φ_i), k given as integer mode numbers.
    PlaneWaves { modes: Vec<([i64; 3], f64, f64)> },
    /// A Gaussian over/underdensity of amplitude A and width σ at the box center.
    GaussianPeak { amp: f64, sigma: f64 },
}

pub fn preset_field(grid: &Grid, preset: &Preset) -> Vec<f64> {
    let n = grid.n;
    let dx = grid.dx();
    let mut f = vec![0.0; grid.size];
    match preset {
        Preset::PlaneWaves { modes } => {
            for idx in 0..grid.size {
                let ijk = grid.unravel(idx);
                let mut v = 0.0;
                for (m, a, ph) in modes {
                    let mut arg = *ph;
                    for d in 0..grid.dim {
                        arg += 2.0 * PI * m[d] as f64 * ijk[d] as f64 / n as f64;
                    }
                    v += a * arg.cos();
                }
                f[idx] = v;
            }
        }
        Preset::GaussianPeak { amp, sigma } => {
            let c = grid.l / 2.0;
            let mut mean = 0.0;
            for idx in 0..grid.size {
                let ijk = grid.unravel(idx);
                let mut r2 = 0.0;
                for d in 0..grid.dim {
                    let x = ijk[d] as f64 * dx - c;
                    r2 += x * x;
                }
                f[idx] = amp * (-r2 / (2.0 * sigma * sigma)).exp();
                mean += f[idx];
            }
            mean /= grid.size as f64;
            for v in f.iter_mut() {
                *v -= mean;
            }
        }
    }
    f
}

/// Apply Gaussian smoothing exp(-k^2 R^2 / 2)... NOTE: we use the *power*
/// convention P → P exp(-k^2 R^2), i.e. the field is filtered by
/// exp(-k^2 R^2 / 2).
pub fn smooth_hat(grid: &Grid, fhat: &mut [C64], r: f64) {
    if r <= 0.0 {
        return;
    }
    for idx in 0..grid.size {
        let k2 = grid.k2(idx);
        fhat[idx] *= (-0.5 * k2 * r * r).exp();
    }
}
