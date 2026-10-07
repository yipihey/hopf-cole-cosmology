//! hcc-core: numerical core for "Hopf–Cole cosmology" — Burgers, heat
//! equation, nLPT phase-space sheets and the Hopf–Cole spectral solver.

pub mod burgers1d;
pub mod fft;
pub mod grid;
pub mod hopfcole;
pub mod ics;
pub mod lpt;
pub mod rng;
pub mod sheet;
pub mod spectra;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use fft::FftEngine;
use grid::Grid;
use ics::{LinearPk, PkShape, Preset};
use lpt::Lpt;
use num_complex::Complex64 as C64;
use spectra::Spectrum;

/// High-level driver shared by native tests and the WASM bindings.
pub struct Cosmo {
    pub grid: Grid,
    pub eng: FftEngine,
    pub pk: Option<LinearPk>,
    /// Linear density δ0 (Fourier, unnormalized forward FFT), already smoothed.
    pub delta0_hat: Vec<C64>,
    /// Lagrangian potential ϕ with ∇²ϕ = δ0.
    pub phi0: Vec<f64>,
    pub lpt: Option<Lpt>,
    pub sigma0: f64,
}

impl Cosmo {
    pub fn new(dim: usize, n: usize, l: f64) -> Self {
        let grid = Grid::new(dim, n, l);
        let eng = FftEngine::new(&grid);
        Cosmo { grid: grid.clone(), eng, pk: None, delta0_hat: vec![C64::new(0.0, 0.0); grid.size], phi0: vec![0.0; grid.size], lpt: None, sigma0: 0.0 }
    }

    fn finish_ic(&mut self, target_sigma: f64) {
        // normalize rms(δ0) = target_sigma (if > 0)
        let mut var = 0.0;
        for v in &self.delta0_hat {
            var += v.norm_sqr();
        }
        var /= (self.grid.size as f64).powi(2);
        let s = var.sqrt();
        if target_sigma > 0.0 && s > 0.0 {
            let f = target_sigma / s;
            for v in self.delta0_hat.iter_mut() {
                *v *= f;
            }
            if let Some(pk) = self.pk.as_mut() {
                pk.amp *= f * f;
            }
            self.sigma0 = target_sigma;
        } else {
            self.sigma0 = s;
        }
        let phih = fft::inv_laplacian_hat(&self.grid, &self.delta0_hat);
        self.phi0 = self.eng.inverse_to_real(phih);
        self.lpt = None;
    }

    /// Gaussian random field with P(k) = amp k^n T² exp(-k²R²), normalized to rms δ0 = sigma.
    pub fn set_ic_gaussian(&mut self, shape: PkShape, r_smooth: f64, seed: u64, sigma: f64) {
        let mut pk = LinearPk { shape, r_smooth, amp: 1.0, k_cut: 0.0 };
        // pre-normalize amp so that the *expected* grid variance is 1, then rescale to the realization
        let gv = ics::grid_variance(&self.grid, &|k| pk.eval(k));
        if gv > 0.0 {
            pk.amp = 1.0 / gv;
        }
        self.delta0_hat = ics::gaussian_field_hat(&self.grid, &mut self.eng, &pk, seed);
        self.pk = Some(pk);
        self.finish_ic(sigma);
    }

    /// Deterministic preset field, smoothed with R, normalized to rms sigma (0 = keep amplitude).
    pub fn set_ic_preset(&mut self, preset: Preset, r_smooth: f64, sigma: f64) {
        let f = ics::preset_field(&self.grid, &preset);
        let mut fh = self.eng.forward_real(&f);
        ics::smooth_hat(&self.grid, &mut fh, r_smooth);
        fh[0] = C64::new(0.0, 0.0);
        self.delta0_hat = fh;
        self.pk = None;
        self.finish_ic(sigma);
    }

    pub fn build_lpt(&mut self, order: usize) {
        let need = match &self.lpt {
            Some(l) => l.order < order,
            None => true,
        };
        if need {
            self.lpt = Some(Lpt::new(&self.grid, &mut self.eng, &self.delta0_hat, order));
        }
    }

    pub fn lpt_ref(&self) -> &Lpt {
        self.lpt.as_ref().expect("call build_lpt first")
    }

    pub fn positions(&self, d: f64, order: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.grid.size * self.grid.dim];
        self.lpt_ref().positions(d, order, &mut out);
        out
    }
    pub fn velocities(&self, d: f64, order: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.grid.size * self.grid.dim];
        self.lpt_ref().velocities(d, order, &mut out);
        out
    }
    pub fn sheet_density(&self, d: f64, order: usize, ne: usize, ss: usize) -> Vec<f32> {
        let pos = self.positions(d, order);
        sheet::sheet_density_2d(&self.grid, &pos, ne, ss)
    }
    pub fn cic_density(&self, d: f64, order: usize, ne: usize) -> Vec<f32> {
        let pos = self.positions(d, order);
        sheet::cic_density(self.grid.dim, &pos, self.grid.size, self.grid.l, ne)
    }
    /// Linear density field D δ0 in real space.
    pub fn linear_delta(&mut self, d: f64) -> Vec<f32> {
        let h: Vec<C64> = self.delta0_hat.iter().map(|v| v * d).collect();
        self.eng.inverse_to_real_f32(h)
    }
    pub fn hopf_cole(&mut self, d: f64, nu: f64, method: hopfcole::HcMethod) -> hopfcole::HopfColeResult {
        hopfcole::hopf_cole_solve(&self.grid, &mut self.eng, &self.phi0, nu, d, method)
    }
    pub fn power_spectrum(&mut self, f: &[f32], nbins: usize, deconvolve_cic: bool) -> Spectrum {
        let fh = self.eng.forward_real_f32(f);
        if deconvolve_cic {
            let g = self.grid.clone();
            let w = move |idx: usize| sheet::cic_window(&g, idx).powi(2);
            spectra::power_spectrum_hat(&self.grid, &fh, &fh, nbins, Some(&w))
        } else {
            spectra::power_spectrum_hat(&self.grid, &fh, &fh, nbins, None)
        }
    }
    pub fn cross_spectrum(&mut self, f: &[f32], g: &[f32], nbins: usize) -> Spectrum {
        let fh = self.eng.forward_real_f32(f);
        let gh = self.eng.forward_real_f32(g);
        spectra::power_spectrum_hat(&self.grid, &fh, &gh, nbins, None)
    }
    pub fn fourier_maps(&mut self, f: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let fh = self.eng.forward_real_f32(f);
        spectra::fourier_maps(&self.grid, &fh)
    }
    /// Linear theory P(k) at D for the model (None for presets).
    pub fn linear_pk(&self, k: f64, d: f64) -> f64 {
        match &self.pk {
            Some(pk) => d * d * pk.eval(k),
            None => 0.0,
        }
    }
    /// 1-loop prediction at D: returns (P_lin, P22, P13) at each k.
    pub fn one_loop(&self, ks: &[f64], d: f64, kernels: spectra::Kernels) -> Vec<[f64; 3]> {
        let pk = match &self.pk {
            Some(p) => p.clone(),
            None => return ks.iter().map(|_| [0.0; 3]).collect(),
        };
        let f = |k: f64| pk.eval(k);
        let qmin = self.grid.kf() * 0.05;
        let qmax = self.grid.knyq() * 4.0;
        ks.iter()
            .map(|&k| {
                let (p22, p13) = spectra::one_loop(&f, self.grid.dim, k, kernels, qmin, qmax, 120, 64);
                [d * d * f(k), d.powi(4) * p22, d.powi(4) * p13]
            })
            .collect()
    }
    pub fn shell_crossing(&self, order: usize) -> f64 {
        if order <= 1 { self.lpt_ref().zeldovich_shell_crossing() } else { self.lpt_ref().shell_crossing(order) }
    }
    pub fn built_order(&self) -> usize {
        self.lpt.as_ref().map(|l| l.order).unwrap_or(0)
    }
    pub fn has_linear_pk(&self) -> bool {
        self.pk.is_some()
    }
    /// Grid estimate of σ_v² = (1/d)(1/L^d) Σ_k P_lin(k)/k² (for the Zel'dovich propagator).
    pub fn sigma_v2(&self) -> f64 {
        let pk = match &self.pk { Some(p) => p, None => return 0.0 };
        let mut s = 0.0;
        for idx in 1..self.grid.size {
            let k2 = self.grid.k2(idx);
            if k2 > 0.0 { s += pk.eval(k2.sqrt()) / k2; }
        }
        s / (self.grid.dim as f64) / self.grid.l.powi(self.grid.dim as i32)
    }
}
