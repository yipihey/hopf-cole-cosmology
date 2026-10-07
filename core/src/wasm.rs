//! wasm-bindgen surface.
use crate::burgers1d;
use crate::hopfcole::{HcMethod, HopfColeResult};
use crate::ics::{PkShape, Preset};
use crate::spectra::Kernels;
use crate::Cosmo;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct CosmoSim {
    inner: Cosmo,
    hc: Option<HopfColeResult>,
}

#[wasm_bindgen]
impl CosmoSim {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, n: usize, l: f64) -> CosmoSim {
        CosmoSim { inner: Cosmo::new(dim, n, l), hc: None }
    }
    pub fn dim(&self) -> usize { self.inner.grid.dim }
    pub fn n(&self) -> usize { self.inner.grid.n }
    pub fn boxsize(&self) -> f64 { self.inner.grid.l }
    pub fn sigma0(&self) -> f64 { self.inner.sigma0 }

    /// shape: 0 = power law (p1 = n), 1 = BBKS (p1 = ns, p2 = Γ), 2 = EH no-wiggle (p1 = ns, p2 = Γ)
    pub fn set_ic_gaussian(&mut self, shape: u32, p1: f64, p2: f64, r_smooth: f64, seed: u32, sigma: f64) {
        let s = match shape {
            0 => PkShape::PowerLaw { n: p1 },
            1 => PkShape::Bbks { ns: p1, gamma: p2 },
            _ => PkShape::EhNoWiggle { ns: p1, gamma: p2 },
        };
        self.inner.set_ic_gaussian(s, r_smooth, seed as u64, sigma);
        self.hc = None;
    }
    /// modes: flat [kx, ky, kz, amp, phase] × m (integer mode numbers)
    pub fn set_ic_plane_waves(&mut self, modes: &[f64], r_smooth: f64, sigma: f64) {
        let mut v = Vec::new();
        for c in modes.chunks(5) {
            v.push(([c[0] as i64, c[1] as i64, c[2] as i64], c[3], c[4]));
        }
        self.inner.set_ic_preset(Preset::PlaneWaves { modes: v }, r_smooth, sigma);
        self.hc = None;
    }
    pub fn set_ic_peak(&mut self, amp: f64, width: f64, r_smooth: f64, sigma: f64) {
        self.inner.set_ic_preset(Preset::GaussianPeak { amp, sigma: width }, r_smooth, sigma);
        self.hc = None;
    }
    pub fn build_lpt(&mut self, order: usize) { self.inner.build_lpt(order); }
    pub fn positions(&self, d: f64, order: usize) -> Vec<f32> { self.inner.positions(d, order) }
    pub fn velocities(&self, d: f64, order: usize) -> Vec<f32> { self.inner.velocities(d, order) }
    pub fn sheet_density(&self, d: f64, order: usize, ne: usize, ss: usize) -> Vec<f32> { self.inner.sheet_density(d, order, ne, ss) }
    pub fn cic_density(&self, d: f64, order: usize, ne: usize) -> Vec<f32> { self.inner.cic_density(d, order, ne) }
    pub fn linear_delta(&mut self, d: f64) -> Vec<f32> { self.inner.linear_delta(d) }
    pub fn phi0(&self) -> Vec<f32> { self.inner.phi0.iter().map(|&v| v as f32).collect() }
    pub fn lpt_psi(&self, order: usize, comp: usize) -> Vec<f32> { self.inner.lpt_ref().psi[order - 1][comp].clone() }
    pub fn lpt_div(&self, order: usize) -> Vec<f32> { self.inner.lpt_ref().div[order - 1].clone() }
    pub fn lpt_curl(&self, order: usize, comp: usize) -> Vec<f32> {
        let l = self.inner.lpt_ref();
        if l.curl[order - 1].is_empty() { vec![0.0; l.grid.size] } else { l.curl[order - 1][comp].clone() }
    }
    pub fn jacobian(&self, d: f64, order: usize) -> Vec<f32> {
        let mut j = vec![0.0f64; self.inner.grid.size];
        self.inner.lpt_ref().jacobian(d, order, &mut j);
        j.iter().map(|&v| v as f32).collect()
    }
    pub fn shell_crossing(&self, order: usize) -> f64 { self.inner.shell_crossing(order) }

    /// method 0 = Fourier multiplier (dynamic range limited to max_exp),
    /// method ≥ 1 = real-space log-domain kernel with refinement factor = method.
    pub fn hopf_cole(&mut self, d: f64, nu: f64, method: u32, max_exp: f64) {
        let m = if method == 0 { HcMethod::Spectral { max_exp } } else { HcMethod::RealSpace { refine: method as usize } };
        self.hc = Some(self.inner.hopf_cole(d, nu, m));
    }
    pub fn hc_delta(&self) -> Vec<f32> { self.hc.as_ref().unwrap().delta.clone() }
    pub fn hc_phi(&self) -> Vec<f32> { self.hc.as_ref().unwrap().phi_v.clone() }
    pub fn hc_velocity(&self) -> Vec<f32> { self.hc.as_ref().unwrap().velocity.clone() }
    pub fn hc_lnpsi(&self) -> Vec<f32> { self.hc.as_ref().unwrap().lnpsi.clone() }
    pub fn hc_psihat_log(&self) -> Vec<f32> { self.hc.as_ref().unwrap().psihat_log.clone() }
    pub fn hc_nu_eff(&self) -> f64 { self.hc.as_ref().unwrap().nu_eff }
    pub fn hc_exponent_range(&self) -> f64 { self.hc.as_ref().unwrap().exponent_range }

    /// Returns flat [k..., P..., N...] (3 × nb entries, nb = number of non-empty bins).
    pub fn power_spectrum(&mut self, f: &[f32], nbins: usize, deconvolve_cic: bool) -> Vec<f64> {
        let s = self.inner.power_spectrum(f, nbins, deconvolve_cic);
        let mut out = s.k.clone();
        out.extend(s.p.iter());
        out.extend(s.nmodes.iter());
        out
    }
    pub fn cross_spectrum(&mut self, f: &[f32], g: &[f32], nbins: usize) -> Vec<f64> {
        let s = self.inner.cross_spectrum(f, g, nbins);
        let mut out = s.k.clone();
        out.extend(s.p.iter());
        out.extend(s.nmodes.iter());
        out
    }
    /// Fourier amplitude map: log10|f̂|/max, fft-shifted n×n (k_z = 0 plane in 3D).
    pub fn fourier_amp(&mut self, f: &[f32]) -> Vec<f32> { self.inner.fourier_maps(f).0 }
    pub fn fourier_phase(&mut self, f: &[f32]) -> Vec<f32> { self.inner.fourier_maps(f).1 }
    pub fn linear_pk(&self, ks: &[f64], d: f64) -> Vec<f64> { ks.iter().map(|&k| self.inner.linear_pk(k, d)).collect() }
    /// Returns flat [Plin..., P22..., P13...] for kernels 0 = SPT, 1 = Zel'dovich.
    pub fn one_loop(&self, ks: &[f64], d: f64, kernels: u32) -> Vec<f64> {
        let kk = if kernels == 0 { Kernels::Spt } else { Kernels::Zeldovich };
        let r = self.inner.one_loop(ks, d, kk);
        let mut out: Vec<f64> = r.iter().map(|v| v[0]).collect();
        out.extend(r.iter().map(|v| v[1]));
        out.extend(r.iter().map(|v| v[2]));
        out
    }
    pub fn kf(&self) -> f64 { self.inner.grid.kf() }
    pub fn knyq(&self) -> f64 { self.inner.grid.knyq() }
}

// ----------------------------------------------------------------- 1D tools

#[wasm_bindgen]
pub fn burgers_rk4(u0: &[f64], l: f64, nu: f64, t: f64, dealias: bool) -> Vec<f64> {
    let mut b = burgers1d::Burgers1D::new(u0, l, nu, dealias);
    b.advance_to(t, 0.4);
    b.u()
}

#[wasm_bindgen]
pub struct HopfCole1DResult {
    r: burgers1d::HopfCole1D,
}
#[wasm_bindgen]
impl HopfCole1DResult {
    pub fn u(&self) -> Vec<f64> { self.r.u.clone() }
    pub fn phi(&self) -> Vec<f64> { self.r.phi.clone() }
    pub fn phi0(&self) -> Vec<f64> { self.r.phi0.clone() }
    pub fn psi(&self) -> Vec<f64> { self.r.psi.clone() }
    pub fn psi0(&self) -> Vec<f64> { self.r.psi0.clone() }
    pub fn psihat_abs(&self) -> Vec<f64> { self.r.psihat_abs.clone() }
    pub fn psihat0_abs(&self) -> Vec<f64> { self.r.psihat0_abs.clone() }
    pub fn uhat_abs(&self) -> Vec<f64> { self.r.uhat_abs.clone() }
    pub fn kernel(&self) -> Vec<f64> { self.r.kernel.clone() }
}
#[wasm_bindgen]
pub fn hopf_cole_1d(u0: &[f64], l: f64, nu: f64, t: f64) -> HopfCole1DResult {
    HopfCole1DResult { r: burgers1d::hopf_cole_1d(u0, l, nu, t) }
}

#[wasm_bindgen]
pub struct Inviscid1DResult {
    r: burgers1d::Inviscid1D,
}
#[wasm_bindgen]
impl Inviscid1DResult {
    pub fn x_char(&self) -> Vec<f64> { self.r.x_char.clone() }
    pub fn u(&self) -> Vec<f64> { self.r.u.clone() }
    pub fn x0_star(&self) -> Vec<f64> { self.r.x0_star.clone() }
    pub fn rho(&self) -> Vec<f64> { self.r.rho.clone() }
    pub fn phi(&self) -> Vec<f64> { self.r.phi.clone() }
}
#[wasm_bindgen]
pub fn inviscid_1d(u0: &[f64], l: f64, t: f64) -> Inviscid1DResult {
    Inviscid1DResult { r: burgers1d::inviscid_1d(u0, l, t) }
}

/// Heat equation on a periodic 1D grid: exact Fourier solution.
#[wasm_bindgen]
pub fn heat_1d(f0: &[f64], l: f64, nu: f64, t: f64) -> Vec<f64> {
    let grid = crate::grid::Grid::new(1, f0.len(), l);
    let mut eng = crate::fft::FftEngine::new(&grid);
    let mut h = eng.forward_real(f0);
    for i in 0..grid.n {
        let k = grid.k1(i);
        h[i] *= (-nu * k * k * t).exp();
    }
    eng.inverse_to_real(h)
}

/// |f̂_k| / N for a real 1D signal.
#[wasm_bindgen]
pub fn spectrum_1d(f: &[f64]) -> Vec<f64> {
    let grid = crate::grid::Grid::new(1, f.len(), 1.0);
    let mut eng = crate::fft::FftEngine::new(&grid);
    let h = eng.forward_real(f);
    h.iter().map(|v| v.norm() / f.len() as f64).collect()
}
