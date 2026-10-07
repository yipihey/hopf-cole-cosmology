//! Hopf–Cole spectral solver for potential Burgers flow in d dimensions.
//!
//!   ∂_D u + (u·∇)u = ν ∇²u,   u = ∇Φ   ⇒   ∂_D Φ + ½|∇Φ|² = ν∇²Φ
//!   ψ = exp(-Φ/2ν)            ⇒   ∂_D ψ = ν ∇²ψ   (linear heat equation)
//!
//! Cosmology: with D the growth factor as time, the Zel'dovich velocity
//! u = Ψ^(1) = -∇ϕ obeys the *inviscid* Burgers equation.  Finite ν gives the
//! adhesion model.  The Eulerian density follows from the inverse Lagrangian
//! map q = x - D u(x,D):  1+δ = det(∂q/∂x) = det(I - D ∇∇Φ)  (single-stream).

use crate::fft::FftEngine;
use crate::grid::Grid;
use crate::lpt::det_i_plus;
use num_complex::Complex64 as C64;

pub struct HopfColeResult {
    pub delta: Vec<f32>,
    pub phi_v: Vec<f32>,
    pub velocity: Vec<f32>,
    /// ln ψ(x,D) (up to a constant)
    pub lnpsi: Vec<f32>,
    /// log10 |ψ̂(k,D)| normalized to its maximum
    pub psihat_log: Vec<f32>,
    /// effective viscosity actually used (after the precision floor)
    pub nu_eff: f64,
    /// dynamic range exponent (ϕ_max-ϕ_min)/(2ν)
    pub exponent_range: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HcMethod {
    /// Fourier multiplier exp(-ν k² D) acting on ψ̂0.  Limited by floating
    /// point dynamic range: the exponent (ϕ_max-ϕ_min)/2ν must stay ≲ max_exp.
    Spectral { max_exp: f64 },
    /// Real-space heat-kernel convolution carried out in the log domain
    /// (separable log-sum-exp).  Stable for any ν down to the grid scale;
    /// reduces to the discrete Hopf–Lax (Lax–Oleinik) formula as ν → 0.
    /// `refine` ≥ 1 Fourier-upsamples ϕ by that factor before the convolution,
    /// lowering the grid floor ν_min ≈ dx²/(4 D refine²).
    RealSpace { refine: usize },
}

/// Fourier (zero-padding) upsampling of a real field by an integer factor r.
pub fn upsample_real(grid: &Grid, eng: &mut FftEngine, f: &[f64], r: usize) -> (Grid, Vec<f64>) {
    if r <= 1 {
        return (grid.clone(), f.to_vec());
    }
    let fh = eng.forward_real(f);
    let n = grid.n;
    let nf = n * r;
    let fine = Grid::new(grid.dim, nf, grid.l);
    let mut fine_eng = FftEngine::new(&fine);
    let mut out = vec![C64::new(0.0, 0.0); fine.size];
    let scale = (fine.size as f64) / (grid.size as f64);
    for idx in 0..grid.size {
        let ijk = grid.unravel(idx);
        let mut fijk = [0usize; 3];
        let mut skip = false;
        for a in 0..grid.dim {
            let fr = grid.ifreq(ijk[a]);
            if n % 2 == 0 && ijk[a] == n / 2 {
                skip = true; // drop the Nyquist mode
            }
            fijk[a] = if fr >= 0 { fr as usize } else { (nf as i64 + fr) as usize };
        }
        if skip {
            continue;
        }
        out[fine.ravel(fijk)] = fh[idx] * scale;
    }
    fine_eng.inverse(&mut out);
    (fine, out.iter().map(|v| v.re).collect())
}

/// Separable log-domain Gaussian convolution:
/// returns ln ∫ G(x-y; νD) exp(a(y)) dy with G the heat kernel of the
/// d-dimensional heat equation at time D (variance 2νD per axis).
pub fn log_heat_convolve(grid: &Grid, a: &[f64], nu: f64, dgrow: f64, w: usize) -> Vec<f64> {
    let n = grid.n;
    let d = grid.dim;
    let dx = grid.dx();
    let mut cur = a.to_vec();
    if nu * dgrow <= 0.0 {
        return cur;
    }
    let w = w.min(n / 2);
    let coef: Vec<f64> = (0..=w).map(|s| -(s as f64 * dx).powi(2) / (4.0 * nu * dgrow)).collect();
    let mut line = vec![0.0f64; n];
    let mut out = vec![0.0f64; n];
    for axis in 0..d {
        let stride = n.pow((d - 1 - axis) as u32);
        let block = stride * n;
        let nblocks = grid.size / block;
        for b in 0..nblocks {
            for s0 in 0..stride {
                let base = b * block + s0;
                for i in 0..n {
                    line[i] = cur[base + i * stride];
                }
                for x in 0..n {
                    // max first for stability
                    let mut m = f64::NEG_INFINITY;
                    for s in 0..=w {
                        let yp = (x + s) % n;
                        let ym = (x + n - s) % n;
                        let v1 = line[yp] + coef[s];
                        let v2 = line[ym] + coef[s];
                        if v1 > m { m = v1; }
                        if v2 > m { m = v2; }
                    }
                    let mut sum = 0.0;
                    for s in 0..=w {
                        let yp = (x + s) % n;
                        let v1 = line[yp] + coef[s] - m;
                        sum += v1.exp();
                        if s > 0 {
                            let ym = (x + n - s) % n;
                            let v2 = line[ym] + coef[s] - m;
                            sum += v2.exp();
                        }
                    }
                    out[x] = m + sum.ln();
                }
                for i in 0..n {
                    cur[base + i * stride] = out[i];
                }
            }
        }
    }
    let norm = (d as f64) * dx.ln() - 0.5 * (d as f64) * (4.0 * std::f64::consts::PI * nu * dgrow).ln();
    for v in cur.iter_mut() {
        *v += norm;
    }
    cur
}

/// `phi` is the Lagrangian potential ϕ (∇²ϕ = δ0), so Φ_v = -ϕ and ψ0 = exp(ϕ/2ν).
pub fn hopf_cole_solve(grid: &Grid, eng: &mut FftEngine, phi: &[f64], nu: f64, dgrow: f64, method: HcMethod) -> HopfColeResult {
    let d = grid.dim;
    let phimax = phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let phimin = phi.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = phimax - phimin;
    let mut nu_eff = nu.max(1e-300);
    let (lnpsi, psihat_log): (Vec<f64>, Vec<f32>) = match method {
        HcMethod::Spectral { max_exp } => {
            if range / (2.0 * nu_eff) > max_exp {
                nu_eff = range / (2.0 * max_exp);
            }
            let mut psihat: Vec<C64> = phi.iter().map(|&p| C64::new(((p - phimax) / (2.0 * nu_eff)).exp(), 0.0)).collect();
            eng.forward(&mut psihat);
            for idx in 0..grid.size {
                psihat[idx] *= (-nu_eff * grid.k2(idx) * dgrow).exp();
            }
            let pl = psihat_log_map(&psihat);
            let psi = eng.inverse_to_real(psihat);
            (psi.iter().map(|&p| p.max(1e-290).ln()).collect(), pl)
        }
        HcMethod::RealSpace { refine } => {
            // window: max displacement + 6 kernel widths
            let phih = eng.forward_real(phi);
            let mut umax2 = 0.0f64;
            let mut grads: Vec<Vec<f64>> = Vec::new();
            for a in 0..d {
                grads.push(crate::fft::gradient_component(eng, &phih, a));
            }
            for idx in 0..grid.size {
                let mut s = 0.0;
                for a in 0..d {
                    s += grads[a][idx] * grads[a][idx];
                }
                umax2 = umax2.max(s);
            }
            let reach = dgrow * umax2.sqrt() + 6.0 * (2.0 * nu_eff * dgrow).sqrt();
            let r = refine.max(1);
            let (fine, phif) = upsample_real(grid, eng, phi, r);
            let w = (reach / fine.dx()).ceil() as usize + 2;
            let a: Vec<f64> = phif.iter().map(|&p| p / (2.0 * nu_eff)).collect();
            let lnfine = log_heat_convolve(&fine, &a, nu_eff, dgrow, w);
            let lnpsi: Vec<f64> = if r == 1 {
                lnfine
            } else {
                (0..grid.size)
                    .map(|idx| {
                        let ijk = grid.unravel(idx);
                        lnfine[fine.ravel([ijk[0] * r, ijk[1] * r, ijk[2] * r])]
                    })
                    .collect()
            };
            // Fourier display map from the (range-limited) exponential
            let lmax = lnpsi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut psihat: Vec<C64> = lnpsi.iter().map(|&l| C64::new((l - lmax).max(-700.0).exp(), 0.0)).collect();
            eng.forward(&mut psihat);
            let pl = psihat_log_map(&psihat);
            (lnpsi, pl)
        }
    };
    let exponent_range = range / (2.0 * nu_eff);
    // Φ_v = -2ν ln ψ ; velocity u = ∇Φ_v ; Hessian H = ∇∇Φ_v.
    // Derivatives are 4th-order central finite differences: they are local, so
    // an under-resolved shock (width ν/Δu) does not produce global Gibbs ringing
    // the way spectral derivatives would.
    let phiv: Vec<f64> = lnpsi.iter().map(|&l| -2.0 * nu_eff * l).collect();
    let mut velocity = vec![0.0f32; grid.size * d];
    let mut grads: Vec<Vec<f64>> = Vec::new();
    for a in 0..d {
        let ga = fd_derivative(grid, &phiv, a);
        for idx in 0..grid.size {
            velocity[idx * d + a] = ga[idx] as f32;
        }
        grads.push(ga);
    }
    let mut hess: Vec<Vec<f64>> = vec![Vec::new(); d * d];
    for a in 0..d {
        for b in a..d {
            let h = if a == b { fd_second_derivative(grid, &phiv, a) } else { fd_derivative(grid, &grads[a], b) };
            hess[a * d + b] = h.clone();
            if a != b {
                hess[b * d + a] = h;
            }
        }
    }
    let mut delta = vec![0.0f32; grid.size];
    let mut mt = vec![0.0; d * d];
    for idx in 0..grid.size {
        for c in 0..d * d {
            mt[c] = -dgrow * hess[c][idx];
        }
        delta[idx] = (det_i_plus(&mt, d) - 1.0) as f32;
    }
    HopfColeResult {
        delta,
        phi_v: phiv.iter().map(|&v| v as f32).collect(),
        velocity,
        lnpsi: lnpsi.iter().map(|&v| v as f32).collect(),
        psihat_log,
        nu_eff,
        exponent_range,
    }
}

/// 4th-order central first derivative along `axis` on the periodic grid.
pub fn fd_derivative(grid: &Grid, f: &[f64], axis: usize) -> Vec<f64> {
    let n = grid.n;
    let d = grid.dim;
    let stride = n.pow((d - 1 - axis) as u32);
    let block = stride * n;
    let h = grid.dx();
    let mut out = vec![0.0; grid.size];
    for idx in 0..grid.size {
        let b = idx / block;
        let within = idx % block;
        let i = within / stride;
        let s0 = within % stride;
        let at = |j: i64| -> f64 { f[b * block + (j.rem_euclid(n as i64) as usize) * stride + s0] };
        let i = i as i64;
        out[idx] = (8.0 * (at(i + 1) - at(i - 1)) - (at(i + 2) - at(i - 2))) / (12.0 * h);
    }
    out
}

/// 4th-order central second derivative along `axis` on the periodic grid.
pub fn fd_second_derivative(grid: &Grid, f: &[f64], axis: usize) -> Vec<f64> {
    let n = grid.n;
    let d = grid.dim;
    let stride = n.pow((d - 1 - axis) as u32);
    let block = stride * n;
    let h2 = grid.dx() * grid.dx();
    let mut out = vec![0.0; grid.size];
    for idx in 0..grid.size {
        let b = idx / block;
        let within = idx % block;
        let i = within / stride;
        let s0 = within % stride;
        let at = |j: i64| -> f64 { f[b * block + (j.rem_euclid(n as i64) as usize) * stride + s0] };
        let i = i as i64;
        out[idx] = (-(at(i + 2) + at(i - 2)) + 16.0 * (at(i + 1) + at(i - 1)) - 30.0 * at(i)) / (12.0 * h2);
    }
    out
}

fn psihat_log_map(psihat: &[C64]) -> Vec<f32> {
    let mut maxabs = 0.0f64;
    for v in psihat.iter() {
        maxabs = maxabs.max(v.norm());
    }
    let maxabs = maxabs.max(1e-300);
    psihat.iter().map(|v| ((v.norm() / maxabs).max(1e-300)).log10() as f32).collect()
}

/// Zero-viscosity limit via the Hopf–Lax (Lax–Oleinik) formula in 1D:
/// Φ(x,t) = min_y [Φ0(y) + (x-y)²/(2t)].  Periodic in L; returns Φ and the
/// minimizer y*(x) (the inverse Lagrangian map).  O(N) lower envelope of
/// parabolas (Felzenszwalb & Huttenlocher 2012).
pub fn hopf_lax_1d(phi0: &[f64], l: f64, t: f64) -> (Vec<f64>, Vec<f64>) {
    let n = phi0.len();
    let dx = l / n as f64;
    // triple the domain for periodicity
    let m = 3 * n;
    let f: Vec<f64> = (0..m).map(|i| 2.0 * t * phi0[i % n]).collect(); // minimize (x-y)^2 + f(y)
    let y: Vec<f64> = (0..m).map(|i| (i as f64 - n as f64) * dx).collect();
    // lower envelope
    let mut v = vec![0usize; m];
    let mut z = vec![0.0f64; m + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..m {
        loop {
            let p = v[k];
            let s = ((f[q] + y[q] * y[q]) - (f[p] + y[p] * y[p])) / (2.0 * y[q] - 2.0 * y[p]);
            if s <= z[k] {
                if k == 0 {
                    // replace
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    let mut phi = vec![0.0; n];
    let mut ystar = vec![0.0; n];
    let mut k = 0usize;
    for i in 0..n {
        let x = i as f64 * dx;
        while z[k + 1] < x {
            k += 1;
        }
        let p = v[k];
        phi[i] = ((x - y[p]).powi(2) + f[p]) / (2.0 * t);
        ystar[i] = y[p];
    }
    (phi, ystar)
}
