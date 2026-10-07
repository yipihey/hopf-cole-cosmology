//! One-dimensional Burgers equation: pseudo-spectral time integration,
//! exact Hopf–Cole solution, and the inviscid (Hopf–Lax) limit.

use crate::fft::FftEngine;
use crate::grid::Grid;
use crate::hopfcole::hopf_lax_1d;
use num_complex::Complex64 as C64;

/// Pseudo-spectral integrating-factor RK4 integrator for u_t + u u_x = ν u_xx.
pub struct Burgers1D {
    pub grid: Grid,
    pub eng: FftEngine,
    pub nu: f64,
    pub t: f64,
    pub uhat: Vec<C64>,
    pub dealias: bool,
}

impl Burgers1D {
    pub fn new(u0: &[f64], l: f64, nu: f64, dealias: bool) -> Self {
        let grid = Grid::new(1, u0.len(), l);
        let mut eng = FftEngine::new(&grid);
        let uhat = eng.forward_real(u0);
        Burgers1D { grid, eng, nu, t: 0.0, uhat, dealias }
    }
    fn nonlinear(&mut self, uhat: &[C64]) -> Vec<C64> {
        // N(û) = -(i k / 2) FFT(u²)
        let u = self.eng.inverse_to_real(uhat.to_vec());
        let u2: Vec<f64> = u.iter().map(|v| v * v).collect();
        let mut u2h = self.eng.forward_real(&u2);
        let n = self.grid.n;
        for i in 0..n {
            let k = self.grid.kd(i);
            if self.dealias && (self.grid.ifreq(i).unsigned_abs() as usize) > n / 3 {
                u2h[i] = C64::new(0.0, 0.0);
            } else {
                u2h[i] *= C64::new(0.0, -0.5 * k);
            }
        }
        u2h
    }
    /// Advance by dt using the integrating factor for the diffusion term.
    pub fn step(&mut self, dt: f64) {
        let n = self.grid.n;
        let e_half: Vec<f64> = (0..n).map(|i| (-self.nu * self.grid.k1(i).powi(2) * dt / 2.0).exp()).collect();
        let e_full: Vec<f64> = e_half.iter().map(|v| v * v).collect();
        let u = self.uhat.clone();
        let a = self.nonlinear(&u);
        let ua: Vec<C64> = (0..n).map(|i| e_half[i] * (u[i] + 0.5 * dt * a[i])).collect();
        let b = self.nonlinear(&ua);
        let ub: Vec<C64> = (0..n).map(|i| e_half[i] * u[i] + 0.5 * dt * b[i]).collect();
        let c = self.nonlinear(&ub);
        let uc: Vec<C64> = (0..n).map(|i| e_half[i] * e_half[i] * u[i] + e_half[i] * dt * c[i]).collect();
        let d = self.nonlinear(&uc);
        for i in 0..n {
            self.uhat[i] = e_full[i] * u[i] + dt / 6.0 * (e_full[i] * a[i] + 2.0 * e_half[i] * (b[i] + c[i]) + d[i]);
        }
        self.t += dt;
    }
    /// Advance to time t_end with an advective CFL constraint.
    pub fn advance_to(&mut self, t_end: f64, cfl: f64) {
        while self.t < t_end - 1e-15 {
            let u = self.eng.inverse_to_real(self.uhat.clone());
            let umax = u.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let dt = (cfl * self.grid.dx() / umax).min(t_end - self.t);
            self.step(dt);
        }
    }
    pub fn u(&mut self) -> Vec<f64> {
        self.eng.inverse_to_real(self.uhat.clone())
    }
}

pub struct HopfCole1D {
    pub u: Vec<f64>,
    pub phi: Vec<f64>,
    pub psi: Vec<f64>,
    pub psi0: Vec<f64>,
    pub psihat_abs: Vec<f64>,
    pub psihat0_abs: Vec<f64>,
    pub uhat_abs: Vec<f64>,
    pub kernel: Vec<f64>,
    pub phi0: Vec<f64>,
}

/// Exact solution of viscous Burgers at time t via Hopf–Cole (periodic box).
/// A nonzero mean velocity is handled by Galilean shift.
pub fn hopf_cole_1d(u0: &[f64], l: f64, nu: f64, t: f64) -> HopfCole1D {
    let n = u0.len();
    let grid = Grid::new(1, n, l);
    let mut eng = FftEngine::new(&grid);
    let ubar = u0.iter().sum::<f64>() / n as f64;
    let up: Vec<f64> = u0.iter().map(|v| v - ubar).collect();
    // Φ0 with Φ0' = u' : Φ̂0 = û/(ik)
    let uph = eng.forward_real(&up);
    let phi0h: Vec<C64> = (0..n)
        .map(|i| {
            let k = grid.kd(i);
            if k == 0.0 { C64::new(0.0, 0.0) } else { uph[i] / C64::new(0.0, k) }
        })
        .collect();
    let phi0 = eng.inverse_to_real(phi0h);
    let pmin = phi0.iter().cloned().fold(f64::INFINITY, f64::min);
    let psi0: Vec<f64> = phi0.iter().map(|p| (-(p - pmin) / (2.0 * nu)).exp()).collect();
    let psih0 = eng.forward_real(&psi0);
    let psihat0_abs: Vec<f64> = psih0.iter().map(|v| v.norm()).collect();
    // log-domain heat-kernel convolution (stable for any ν); full window
    let a: Vec<f64> = phi0.iter().map(|p| -(p - pmin) / (2.0 * nu)).collect();
    let lnpsi = crate::hopfcole::log_heat_convolve(&grid, &a, nu, t, n / 2);
    let lmax = lnpsi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let psi: Vec<f64> = lnpsi.iter().map(|v| (v - lmax).exp()).collect();
    let psih = eng.forward_real(&psi);
    let psihat_abs: Vec<f64> = psih.iter().map(|v| v.norm()).collect();
    let phi: Vec<f64> = lnpsi.iter().map(|v| -2.0 * nu * (v - lmax) + pmin).collect();
    let phih = eng.forward_real(&phi);
    // u' = Φ_x, then shift x → x - ū t  (multiply by e^{-ik ū t})
    let mut uh: Vec<C64> = (0..n)
        .map(|i| {
            let k = grid.kd(i);
            C64::new(0.0, k) * phih[i] * C64::new(0.0, -k * ubar * t).exp()
        })
        .collect();
    uh[0] = C64::new(ubar * n as f64, 0.0);
    let uhat_abs: Vec<f64> = uh.iter().map(|v| v.norm() / n as f64).collect();
    let u = eng.inverse_to_real(uh);
    // heat kernel on the periodic grid (image sum), centred at x = 0 in index space
    let kernel: Vec<f64> = (0..n)
        .map(|i| {
            let x = grid.k1(0) * 0.0 + (grid.ifreq(i) as f64) * grid.dx();
            if t <= 0.0 {
                if i == 0 { 1.0 / grid.dx() } else { 0.0 }
            } else {
                let s = 4.0 * nu * t;
                let mut g = 0.0;
                for im in -3..=3 {
                    let xx = x + im as f64 * l;
                    g += (-(xx * xx) / s).exp();
                }
                g / (std::f64::consts::PI * s).sqrt()
            }
        })
        .collect();
    HopfCole1D { u, phi, psi, psi0, psihat_abs, psihat0_abs, uhat_abs, kernel, phi0 }
}

pub struct Inviscid1D {
    /// multi-valued characteristic map: x(x0) = x0 + u0(x0) t (unwrapped)
    pub x_char: Vec<f64>,
    /// single-valued entropy solution u(x)
    pub u: Vec<f64>,
    /// inverse Lagrangian map x0*(x) from Hopf–Lax
    pub x0_star: Vec<f64>,
    /// Eulerian density ρ/ρ̄ = ∂x0*/∂x (one-sided finite differences, Lagrangian cells)
    pub rho: Vec<f64>,
    pub phi: Vec<f64>,
}

/// Inviscid Burgers at time t: characteristics plus the Hopf–Lax entropy solution.
pub fn inviscid_1d(u0: &[f64], l: f64, t: f64) -> Inviscid1D {
    let n = u0.len();
    let grid = Grid::new(1, n, l);
    let mut eng = FftEngine::new(&grid);
    let dx = grid.dx();
    let ubar = u0.iter().sum::<f64>() / n as f64;
    let up: Vec<f64> = u0.iter().map(|v| v - ubar).collect();
    let uph = eng.forward_real(&up);
    let phi0h: Vec<C64> = (0..n)
        .map(|i| {
            let k = grid.kd(i);
            if k == 0.0 { C64::new(0.0, 0.0) } else { uph[i] / C64::new(0.0, k) }
        })
        .collect();
    let phi0 = eng.inverse_to_real(phi0h);
    let x_char: Vec<f64> = (0..n).map(|i| i as f64 * dx + u0[i] * t).collect();
    if t <= 0.0 {
        return Inviscid1D { x_char, u: u0.to_vec(), x0_star: (0..n).map(|i| i as f64 * dx).collect(), rho: vec![1.0; n], phi: phi0 };
    }
    let (phi_s, ystar0) = hopf_lax_1d(&phi0, l, t);
    // refine the grid-quantized minimizer with one Newton step on the
    // quadratically interpolated Φ0:  Φ0'(y) - (x - y)/t = 0
    let mut ystar = vec![0.0; n];
    for i in 0..n {
        let x = i as f64 * dx;
        let yj = ystar0[i];
        let j = (yj / dx).round() as i64;
        let ju = j.rem_euclid(n as i64) as usize;
        let jp = (ju + 1) % n;
        let jm = (ju + n - 1) % n;
        let d1 = (phi0[jp] - phi0[jm]) / (2.0 * dx);
        let d2 = (phi0[jp] - 2.0 * phi0[ju] + phi0[jm]) / (dx * dx);
        let g = d1 - (x - yj) / t;
        let h = d2 + 1.0 / t;
        let mut y = yj;
        if h > 0.0 {
            let step = -g / h;
            if step.abs() <= dx { y = yj + step; }
        }
        ystar[i] = y;
    }
    let mut u = vec![0.0; n];
    let mut rho = vec![0.0; n];
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        let mut dy = ystar[ip] - ystar[im];
        if dy > l / 2.0 { dy -= l; }
        if dy < -l / 2.0 { dy += l; }
        rho[i] = (dy / (2.0 * dx)).max(0.0);
        let x = i as f64 * dx;
        let mut dxy = x - ystar[i];
        if dxy > l / 2.0 { dxy -= l; }
        if dxy < -l / 2.0 { dxy += l; }
        u[i] = dxy / t + ubar;
    }
    Inviscid1D { x_char, u, x0_star: ystar, rho, phi: phi_s }
}
