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
    /// Highest LPT order built so far (0 = none).
    pub fn built_order(&self) -> usize { self.inner.built_order() }
    pub fn has_hc(&self) -> bool { self.hc.is_some() }
    pub fn has_linear_pk(&self) -> bool { self.inner.has_linear_pk() }
    pub fn sigma_v2(&self) -> f64 { self.inner.sigma_v2() }
    fn ok_order(&self, order: usize) -> bool { order >= 1 && order <= self.inner.built_order() }
    pub fn positions(&self, d: f64, order: usize) -> Vec<f32> { if !self.ok_order(1) { return vec![]; } self.inner.positions(d, order) }
    pub fn velocities(&self, d: f64, order: usize) -> Vec<f32> { if !self.ok_order(1) { return vec![]; } self.inner.velocities(d, order) }
    /// Phase-space sheet density: triangles in 2D, Kuhn tetrahedra in 3D.
    pub fn sheet_density(&self, d: f64, order: usize, ne: usize, ss: usize) -> Vec<f32> {
        if !self.ok_order(1) { return vec![]; }
        self.inner.sheet_density(d, order, ne, ss)
    }
    pub fn cic_density(&self, d: f64, order: usize, ne: usize) -> Vec<f32> { if !self.ok_order(1) { return vec![]; } self.inner.cic_density(d, order, ne) }
    pub fn linear_delta(&mut self, d: f64) -> Vec<f32> { self.inner.linear_delta(d) }
    pub fn phi0(&self) -> Vec<f32> { self.inner.phi0.iter().map(|&v| v as f32).collect() }
    /// Displacement contribution of the given order at time d, one component:
    /// Σ_{τ∈order} g_τ(d) S^τ_comp / d^order (shape field; equals Ψ^(order) in EdS).
    pub fn lpt_psi(&self, order: usize, comp: usize) -> Vec<f32> {
        if !self.ok_order(order) || comp >= self.inner.grid.dim { return vec![]; }
        let l = self.inner.lpt_ref();
        let mut out = vec![0.0f32; l.grid.size];
        for (t, term) in l.terms.iter().enumerate() {
            if term.spec.order != order { continue; }
            let w = l.growth.g_and_dg(t, 1.0).0 as f32;
            for idx in 0..l.grid.size { out[idx] += w * term.psi[comp][idx]; }
        }
        out
    }
    /// Longitudinal source of the given order (shape at D = 1; see Lpt::div_source).
    pub fn lpt_div(&self, order: usize) -> Vec<f32> { if !self.ok_order(order) { return vec![]; } self.inner.lpt_ref().div_source(order, 1.0) }
    pub fn lpt_curl(&self, order: usize, comp: usize) -> Vec<f32> { if !self.ok_order(order) { return vec![]; } self.inner.lpt_ref().curl_source(order, comp, 1.0) }
    /// Flat ΛCDM cosmology (Ω_m = 1: Einstein–de Sitter, exact D^n growth).
    pub fn set_cosmology(&mut self, omega_m: f64) { self.inner.set_cosmology(omega_m); }
    pub fn omega_m(&self) -> f64 { self.inner.cosmo.omega_m }
    /// Scale factor at linear growth factor d (= d in EdS; D normalized to a at early times).
    pub fn a_of_d(&self, d: f64) -> f64 { self.inner.lpt.as_ref().map(|l| l.growth.a_of_d(d)).unwrap_or(d) }
    pub fn d_of_a(&self, a: f64) -> f64 { self.inner.lpt.as_ref().map(|l| l.growth.d_of_a(a)).unwrap_or(a) }
    /// Largest reachable D (∞ in EdS).
    pub fn d_max(&self) -> f64 { self.inner.lpt.as_ref().map(|l| l.growth.d_max()).unwrap_or(f64::INFINITY) }
    /// Growth-term bookkeeping: labels, and g_τ(d)/d^order for each term.
    /// Raw spatial field S^τ (component) of term τ; positions = q + Σ_τ g_τ(D) S^τ with g from term_growth.
    pub fn term_psi(&self, term: usize, comp: usize) -> Vec<f32> {
        self.inner.lpt.as_ref().and_then(|l| l.terms.get(term).map(|t| t.psi[comp].clone())).unwrap_or_default()
    }
    /// g_τ(D) itself (not divided by D^n).
    pub fn term_g(&self, d: f64) -> Vec<f64> {
        self.inner.lpt.as_ref().map(|l| (0..l.terms.len()).map(|t| l.growth.g_and_dg(t, d).0).collect()).unwrap_or_default()
    }
    pub fn term_labels(&self) -> Vec<String> { self.inner.lpt.as_ref().map(|l| l.term_labels()).unwrap_or_default() }
    pub fn term_orders(&self) -> Vec<u32> { self.inner.lpt.as_ref().map(|l| l.terms.iter().map(|t| t.spec.order as u32).collect()).unwrap_or_default() }
    pub fn term_growth(&self, d: f64) -> Vec<f64> {
        self.inner.lpt.as_ref().map(|l| (0..l.terms.len()).map(|t| l.growth.g_and_dg(t, d).0 / d.max(1e-12).powi(l.terms[t].spec.order as i32)).collect()).unwrap_or_default()
    }
    pub fn jacobian(&self, d: f64, order: usize) -> Vec<f32> {
        if !self.ok_order(1) { return vec![]; }
        let mut j = vec![0.0f64; self.inner.grid.size];
        self.inner.lpt_ref().jacobian(d, order, &mut j);
        j.iter().map(|&v| v as f32).collect()
    }
    pub fn shell_crossing(&self, order: usize) -> f64 { if !self.ok_order(1) { return f64::INFINITY; } self.inner.shell_crossing(order) }

    /// method 0 = Fourier multiplier (dynamic range limited to max_exp),
    /// method ≥ 1 = real-space log-domain kernel with refinement factor = method.
    pub fn hopf_cole(&mut self, d: f64, nu: f64, method: u32, max_exp: f64) {
        let m = if method == 0 { HcMethod::Spectral { max_exp } } else { HcMethod::RealSpace { refine: method as usize } };
        self.hc = Some(self.inner.hopf_cole(d, nu, m));
    }
    pub fn hc_delta(&self) -> Vec<f32> { self.hc.as_ref().map(|h| h.delta.clone()).unwrap_or_default() }
    pub fn hc_phi(&self) -> Vec<f32> { self.hc.as_ref().map(|h| h.phi_v.clone()).unwrap_or_default() }
    pub fn hc_velocity(&self) -> Vec<f32> { self.hc.as_ref().map(|h| h.velocity.clone()).unwrap_or_default() }
    pub fn hc_lnpsi(&self) -> Vec<f32> { self.hc.as_ref().map(|h| h.lnpsi.clone()).unwrap_or_default() }
    /// fft-shifted n×n map of log10 |ψ̂|/max (same layout as fourier_amp).
    pub fn hc_psihat_log(&self) -> Vec<f32> { self.hc.as_ref().map(|h| h.psihat_log.clone()).unwrap_or_default() }
    pub fn hc_nu_eff(&self) -> f64 { self.hc.as_ref().map(|h| h.nu_eff).unwrap_or(f64::NAN) }
    pub fn hc_exponent_range(&self) -> f64 { self.hc.as_ref().map(|h| h.exponent_range).unwrap_or(f64::NAN) }
    /// Grid floor ν_min ≈ dx²/(4 D refine²) of the real-space method (0 for the Fourier method).
    pub fn hc_nu_floor(&self) -> f64 { self.hc.as_ref().map(|h| h.nu_floor).unwrap_or(f64::NAN) }

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
    /// Both maps from one FFT: [amp (n*n)..., phase (n*n)...].
    pub fn fourier_maps(&mut self, f: &[f32]) -> Vec<f32> {
        let (a, p) = self.inner.fourier_maps(f);
        let mut out = a; out.extend(p); out
    }
    /// Auto spectra of f and g and their cross spectrum from two FFTs:
    /// flat [k..., Pff..., Pgg..., Pfg..., N...] (5 × nb).
    pub fn auto_cross_spectra(&mut self, f: &[f32], g: &[f32], nbins: usize) -> Vec<f64> {
        let fh = self.inner.eng.forward_real_f32(f);
        let gh = self.inner.eng.forward_real_f32(g);
        let sff = crate::spectra::power_spectrum_hat(&self.inner.grid, &fh, &fh, nbins, None);
        let sgg = crate::spectra::power_spectrum_hat(&self.inner.grid, &gh, &gh, nbins, None);
        let sfg = crate::spectra::power_spectrum_hat(&self.inner.grid, &fh, &gh, nbins, None);
        let mut out = sff.k.clone();
        out.extend(sff.p.iter()); out.extend(sgg.p.iter()); out.extend(sfg.p.iter()); out.extend(sff.nmodes.iter());
        out
    }
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

/// Spectral derivative df/dx on a periodic 1D grid of length l.
#[wasm_bindgen]
pub fn derivative_1d(f: &[f64], l: f64) -> Vec<f64> {
    let grid = crate::grid::Grid::new(1, f.len(), l);
    let mut eng = crate::fft::FftEngine::new(&grid);
    let h = eng.forward_real(f);
    crate::fft::gradient_component(&mut eng, &h, 0)
}

/// Zel'dovich velocity u0 = -dϕ/dx with ϕ'' = δ0 (periodic, mean removed).
#[wasm_bindgen]
pub fn zeldovich_velocity_1d(delta0: &[f64], l: f64) -> Vec<f64> {
    let grid = crate::grid::Grid::new(1, delta0.len(), l);
    let mut eng = crate::fft::FftEngine::new(&grid);
    let dh = eng.forward_real(delta0);
    let ph = crate::fft::inv_laplacian_hat(&grid, &dh);
    let g = crate::fft::gradient_component(&mut eng, &ph, 0);
    g.iter().map(|v| -v).collect()
}

/// Multi-stream 1D sheet density (see burgers1d::sheet_density_1d).
#[wasm_bindgen]
pub fn sheet_density_1d(x: &[f64], l: f64, ne: usize) -> Vec<f64> {
    burgers1d::sheet_density_1d(x, l, ne)
}
