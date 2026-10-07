//! hcc-core: numerical core for "Hopf–Cole cosmology" — Burgers, heat
//! equation, nLPT phase-space sheets and the Hopf–Cole spectral solver.

pub mod burgers1d;
pub mod fft;
pub mod grid;
pub mod growth;
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
    pub cosmo: growth::Cosmology,
    /// Keep spectral gradient tensors of all LPT terms (memory heavy in 3D).
    pub keep_tensors: Option<bool>,
}

impl Cosmo {
    pub fn new(dim: usize, n: usize, l: f64) -> Self {
        let grid = Grid::new(dim, n, l);
        let eng = FftEngine::new(&grid);
        Cosmo { grid: grid.clone(), eng, pk: None, delta0_hat: vec![C64::new(0.0, 0.0); grid.size], phi0: vec![0.0; grid.size], lpt: None, sigma0: 0.0, cosmo: growth::Cosmology::eds(), keep_tensors: None }
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

    /// Set the cosmology (flat ΛCDM; Ω_m = 1 is EdS).  Invalidates the LPT fields.
    pub fn set_cosmology(&mut self, omega_m: f64) {
        let c = growth::Cosmology { omega_m: omega_m.clamp(0.05, 1.0) };
        if c != self.cosmo { self.cosmo = c; self.lpt = None; }
    }

    pub fn build_lpt(&mut self, order: usize) {
        let need = match &self.lpt {
            Some(l) => l.order < order || l.cosmo != self.cosmo,
            None => true,
        };
        if need {
            // keep the spectral gradient tensors in 2D (cheap); use FD Jacobians in 3D
            let keep = self.keep_tensors.unwrap_or(self.grid.dim == 2);
            self.lpt = Some(Lpt::new(&self.grid, &mut self.eng, &self.delta0_hat, order, self.cosmo, keep));
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
        if self.grid.dim == 3 { sheet::sheet_density_3d(&self.grid, &pos, ne, ss) } else { sheet::sheet_density_2d(&self.grid, &pos, ne, ss) }
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

    /// Helmholtz decomposition of the nLPT displacement at D:
    /// returns (ϕ_eff, Ψ_T) with Ψ_L = -D ∇ϕ_eff the longitudinal part and Ψ_T the
    /// transverse remainder (interleaved).  For order 1, ϕ_eff = ϕ0 exactly.
    pub fn lpt_potential(&mut self, d: f64, order: usize) -> (Vec<f64>, Vec<f32>) {
        let dim = self.grid.dim;
        let size = self.grid.size;
        let lpt = self.lpt.as_ref().expect("call build_lpt first");
        let mut comps: Vec<Vec<f32>> = Vec::new();
        for a in 0..dim { comps.push(lpt.displacement_component(d, order, a)); }
        // divergence in Fourier space, then ϕ_eff = -∇^{-2}(∇·Ψ)/D
        let mut divhat = vec![C64::new(0.0, 0.0); size];
        let mut hats: Vec<Vec<C64>> = Vec::new();
        for a in 0..dim {
            let h = self.eng.forward_real_f32(&comps[a]);
            for idx in 0..size { let k = self.grid.kvec(idx); divhat[idx] += C64::new(0.0, k[a]) * h[idx]; }
            hats.push(h);
        }
        let dd = if d.abs() < 1e-12 { 1.0 } else { d };
        let phihat: Vec<C64> = (0..size).map(|idx| { let k2 = self.grid.k2(idx); if k2 == 0.0 { C64::new(0.0, 0.0) } else { divhat[idx] / k2 / dd } }).collect();
        let phi_eff = self.eng.inverse_to_real(phihat.clone());
        // transverse part: Ψ_T = Ψ - ∇S with Ŝ = -div̂/k²  (i.e. Ψ̂_L,a = -i k_a div̂ / k²)
        let mut psi_t = vec![0.0f32; size * dim];
        for a in 0..dim {
            let lh: Vec<C64> = (0..size).map(|idx| { let k2 = self.grid.k2(idx); if k2 == 0.0 { C64::new(0.0, 0.0) } else { let k = self.grid.kvec(idx); C64::new(0.0, -k[a] / k2) * divhat[idx] } }).collect();
            let l = self.eng.inverse_to_real(lh);
            for idx in 0..size { psi_t[idx * dim + a] = comps[a][idx] - l[idx] as f32; }
        }
        (phi_eff, psi_t)
    }

    /// Hopf–Cole / Legendre-transform inversion of the nLPT map: the longitudinal
    /// displacement potential of the chosen order replaces the Zel'dovich potential.
    /// With `transverse` the inverse map is corrected to first order in Ψ_T,
    /// q(x) ≈ q_L(x) - Ψ_T(q_L(x)), and the density recomputed from its Jacobian.
    /// Returns the result and rms|Ψ_T| / rms|Ψ_L|.
    pub fn hopf_cole_lpt(&mut self, d: f64, order: usize, nu: f64, method: hopfcole::HcMethod, transverse: bool) -> (hopfcole::HopfColeResult, f64) {
        let dim = self.grid.dim;
        let size = self.grid.size;
        let (phi_eff, psi_t) = self.lpt_potential(d, order);
        let mut r = hopfcole::hopf_cole_solve(&self.grid, &mut self.eng, &phi_eff, nu, d, method);
        // transverse fraction
        let mut st = 0.0; let mut sl = 0.0;
        {
            let lpt = self.lpt.as_ref().unwrap();
            let comps: Vec<Vec<f32>> = (0..dim).map(|a| lpt.displacement_component(d, order, a)).collect();
            for idx in 0..size { for a in 0..dim { let t = psi_t[idx * dim + a] as f64; let full = comps[a][idx] as f64; st += t * t; sl += (full - t) * (full - t); } }
        }
        let frac = if sl > 0.0 { (st / sl).sqrt() } else { 0.0 };
        if transverse && frac > 0.0 {
            // q_L(x) = x - D u(x);  q = q_L - Ψ_T(q_L) (periodic multilinear interpolation of Ψ_T)
            let dx = self.grid.dx();
            let n = self.grid.n;
            let mut q = vec![0.0f64; size * dim];
            let mut i0 = [0usize; 3];
            let mut w1 = [0.0f64; 3];
            for idx in 0..size {
                let ijk = self.grid.unravel(idx);
                let mut ql = [0.0f64; 3];
                for a in 0..dim { ql[a] = ijk[a] as f64 * dx - d * r.velocity[idx * dim + a] as f64; }
                for a in 0..dim { let x = ql[a] / dx; let f = x.floor(); w1[a] = x - f; i0[a] = (f as i64).rem_euclid(n as i64) as usize; }
                let mut t = [0.0f64; 3];
                for c in 0..(1usize << dim) {
                    let mut w = 1.0; let mut jdx = 0usize;
                    for a in 0..dim { let bit = (c >> a) & 1; let ia = (i0[a] + bit) % n; w *= if bit == 1 { w1[a] } else { 1.0 - w1[a] }; jdx = jdx * n + ia; }
                    for a in 0..dim { t[a] += w * psi_t[jdx * dim + a] as f64; }
                }
                for a in 0..dim { q[idx * dim + a] = ql[a] - t[a]; }
            }
            // Jacobian det(∂q/∂x) by 4th-order FD of the displacement (q - x is periodic)
            let mut disp: Vec<Vec<f64>> = vec![vec![0.0; size]; dim];
            for idx in 0..size { let ijk = self.grid.unravel(idx); for a in 0..dim { disp[a][idx] = q[idx * dim + a] - ijk[a] as f64 * dx; } }
            let mut grads: Vec<Vec<f64>> = Vec::new();
            for a in 0..dim { for b in 0..dim { grads.push(hopfcole::fd_derivative(&self.grid, &disp[a], b)); } }
            let mut mt = vec![0.0; dim * dim];
            for idx in 0..size {
                for c in 0..dim * dim { mt[c] = grads[c][idx]; }
                r.delta[idx] = (lpt::det_i_plus(&mt, dim) - 1.0) as f32;
            }
        }
        (r, frac)
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
        // Loop integrals over the modes present in the periodic box only (q ≥ k_f):
        // for red spectra the IR part cancels between P22 and P13 only when both are
        // integrated over the same domain, and the simulation has no modes below k_f.
        let qmin = self.grid.kf() * 0.9;
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
