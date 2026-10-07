//! Lagrangian perturbation theory to arbitrary order (EdS time dependence),
//! in 2D (= 3D with translational symmetry along z) and 3D.
//!
//! Conventions
//! -----------
//! x(q, D) = q + Σ_n D^n Ψ^(n)(q),   δ_lin(D) = D δ0,   Ψ^(1) = -∇ϕ,  ∇²ϕ = δ0.
//!
//! Recursion (Rampf & Hahn 2021, arXiv:2010.12584; verified against an
//! independent cofactor-expansion derivation of the Lagrangian Poisson and
//! Cauchy-invariant equations, see docs):
//!
//!   ∇·Ψ^(n) = Σ_{a+b=n} c2(n;a,b) μ2(M^a, M^b) + Σ_{a+b+c=n} c3(n;a,b,c) μ3(M^a,M^b,M^c)
//!   c2 = [(3-n)/2 - a² - b²] / [(n+3/2)(n-1)]
//!   c3 = [(3-n)/2 - a² - b² - c²] / [(n+3/2)(n-1)]
//!   (∇×Ψ^(n))_i = -(1/n) Σ_{m=1}^{n-1} m ε_{ijk} M^(m)_{lj} M^(n-m)_{lk}
//!
//! with M^(n)_{ab} = ∂_b Ψ^(n)_a, sums over *ordered* tuples, and
//!   μ2(A,B) = ½[trA trB - tr(AB)],
//!   μ3(A,B,C) = (1/6)[trA trB trC - trA tr(BC) - trB tr(AC) - trC tr(AB) + tr(ABC) + tr(ACB)].

use crate::fft::FftEngine;
use crate::grid::Grid;
use num_complex::Complex64 as C64;

pub struct Lpt {
    pub grid: Grid,
    pub order: usize,
    /// psi[n-1][a] : a-th component of Ψ^(n) on the Lagrangian grid.
    pub psi: Vec<Vec<Vec<f32>>>,
    /// m[n-1][a*dim+b] : ∂_b Ψ^(n)_a.
    pub m: Vec<Vec<Vec<f32>>>,
    /// Longitudinal source ∇·Ψ^(n) for n ≥ 2 (n=1: -δ0), kept for pedagogy.
    pub div: Vec<Vec<f32>>,
    /// Transverse source (curl) z-component (2D) or 3 components (3D), n ≥ 3.
    pub curl: Vec<Vec<Vec<f32>>>,
}

#[inline]
fn tr(a: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += a[i * d + i];
    }
    s
}
#[inline]
fn tr_ab(a: &[f64], b: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        for j in 0..d {
            s += a[i * d + j] * b[j * d + i];
        }
    }
    s
}
#[inline]
fn tr_abc(a: &[f64], b: &[f64], c: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                s += a[i * d + j] * b[j * d + k] * c[k * d + i];
            }
        }
    }
    s
}
#[inline]
pub fn mu2(a: &[f64], b: &[f64], d: usize) -> f64 {
    0.5 * (tr(a, d) * tr(b, d) - tr_ab(a, b, d))
}
#[inline]
pub fn mu3(a: &[f64], b: &[f64], c: &[f64], d: usize) -> f64 {
    (tr(a, d) * tr(b, d) * tr(c, d) - tr(a, d) * tr_ab(b, c, d) - tr(b, d) * tr_ab(a, c, d)
        - tr(c, d) * tr_ab(a, b, d) + tr_abc(a, b, c, d) + tr_abc(a, c, b, d)) / 6.0
}
/// det(I + M) for d = 2 or 3.
#[inline]
pub fn det_i_plus(m: &[f64], d: usize) -> f64 {
    if d == 2 {
        (1.0 + m[0]) * (1.0 + m[3]) - m[1] * m[2]
    } else if d == 3 {
        let a = [1.0 + m[0], m[1], m[2], m[3], 1.0 + m[4], m[5], m[6], m[7], 1.0 + m[8]];
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
    } else {
        1.0 + m[0]
    }
}

impl Lpt {
    /// Build the LPT fields from the Fourier-space linear density δ0_hat.
    pub fn new(grid: &Grid, eng: &mut FftEngine, delta0_hat: &[C64], order: usize) -> Self {
        let d = grid.dim;
        let mut lpt = Lpt {
            grid: grid.clone(),
            order,
            psi: Vec::new(),
            m: Vec::new(),
            div: Vec::new(),
            curl: Vec::new(),
        };
        // ---- first order: Ψ̂_a = i k_a δ0_hat / k²
        let mut psihat: Vec<Vec<C64>> = Vec::new();
        for a in 0..d {
            let v: Vec<C64> = (0..grid.size)
                .map(|idx| {
                    let k2 = grid.k2(idx);
                    if k2 == 0.0 {
                        C64::new(0.0, 0.0)
                    } else {
                        let k = grid.kvec(idx);
                        C64::new(0.0, k[a] / k2) * delta0_hat[idx]
                    }
                })
                .collect();
            psihat.push(v);
        }
        let div1: Vec<f32> = eng.inverse_to_real_f32(delta0_hat.iter().map(|v| -v).collect());
        lpt.div.push(div1);
        lpt.curl.push(vec![]);
        lpt.push_order_from_hat(eng, &psihat);

        for n in 2..=order {
            let (s, t) = lpt.sources(n);
            // Fourier-space solve
            let shat = eng.forward_real(&s);
            let that: Vec<Vec<C64>> = t.iter().map(|tc| eng.forward_real(tc)).collect();
            let mut psihat: Vec<Vec<C64>> = vec![vec![C64::new(0.0, 0.0); grid.size]; d];
            for idx in 0..grid.size {
                let k2 = grid.k2(idx);
                if k2 == 0.0 {
                    continue;
                }
                let k = grid.kvec(idx);
                // longitudinal: -i k S/k²
                for a in 0..d {
                    psihat[a][idx] = C64::new(0.0, -k[a] / k2) * shat[idx];
                }
                // transverse: i (k × T)/k²
                if d == 2 {
                    let tz = that[0][idx];
                    psihat[0][idx] += C64::new(0.0, k[1] / k2) * tz;
                    psihat[1][idx] += C64::new(0.0, -k[0] / k2) * tz;
                } else if d == 3 && !that.is_empty() {
                    let (tx, ty, tz) = (that[0][idx], that[1][idx], that[2][idx]);
                    psihat[0][idx] += C64::new(0.0, 1.0 / k2) * (k[1] * tz - k[2] * ty);
                    psihat[1][idx] += C64::new(0.0, 1.0 / k2) * (k[2] * tx - k[0] * tz);
                    psihat[2][idx] += C64::new(0.0, 1.0 / k2) * (k[0] * ty - k[1] * tx);
                }
            }
            lpt.div.push(s.iter().map(|&v| v as f32).collect());
            lpt.curl.push(t.iter().map(|tc| tc.iter().map(|&v| v as f32).collect()).collect());
            lpt.push_order_from_hat(eng, &psihat);
        }
        lpt
    }

    fn push_order_from_hat(&mut self, eng: &mut FftEngine, psihat: &[Vec<C64>]) {
        let d = self.grid.dim;
        let grid = self.grid.clone();
        let mut psi = Vec::new();
        let mut m = vec![Vec::new(); d * d];
        for a in 0..d {
            psi.push(eng.inverse_to_real_f32(psihat[a].clone()));
            for b in 0..d {
                let dh: Vec<C64> = psihat[a]
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| {
                        let k = grid.kvec(idx);
                        C64::new(0.0, k[b]) * v
                    })
                    .collect();
                m[a * d + b] = eng.inverse_to_real_f32(dh);
            }
        }
        self.psi.push(psi);
        self.m.push(m);
    }

    /// Gather the gradient tensor of order n (1-based) at grid index idx.
    #[inline]
    fn tensor_at(&self, n: usize, idx: usize, out: &mut [f64]) {
        let d = self.grid.dim;
        let mn = &self.m[n - 1];
        for c in 0..d * d {
            out[c] = mn[c][idx] as f64;
        }
    }

    /// Longitudinal and transverse sources at order n.
    fn sources(&self, n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let d = self.grid.dim;
        let size = self.grid.size;
        let nf = n as f64;
        let denom = (nf + 1.5) * (nf - 1.0);
        let mut s = vec![0.0; size];
        let ncurl = if d == 2 { 1 } else { 3 };
        let mut t = vec![vec![0.0; size]; ncurl];
        let mut ta = vec![0.0; 9];
        let mut tb = vec![0.0; 9];
        let mut tc = vec![0.0; 9];
        for idx in 0..size {
            let mut sv = 0.0;
            // μ2 pairs (ordered)
            for a in 1..n {
                let b = n - a;
                let c2 = ((3.0 - nf) / 2.0 - (a * a + b * b) as f64) / denom;
                self.tensor_at(a, idx, &mut ta);
                self.tensor_at(b, idx, &mut tb);
                sv += c2 * mu2(&ta[..d * d], &tb[..d * d], d);
            }
            // μ3 triples (ordered), only in 3D
            if d == 3 && n >= 3 {
                for a in 1..n {
                    for b in 1..(n - a) {
                        let c = n - a - b;
                        if c < 1 {
                            continue;
                        }
                        let c3 = ((3.0 - nf) / 2.0 - (a * a + b * b + c * c) as f64) / denom;
                        self.tensor_at(a, idx, &mut ta);
                        self.tensor_at(b, idx, &mut tb);
                        self.tensor_at(c, idx, &mut tc);
                        sv += c3 * mu3(&ta[..9], &tb[..9], &tc[..9], 3);
                    }
                }
            }
            s[idx] = sv;
            // curl: T_i = -(1/n) Σ_m m ε_{ijk} M^(m)_{lj} M^(n-m)_{lk}
            if n >= 3 {
                for mm in 1..n {
                    self.tensor_at(mm, idx, &mut ta);
                    self.tensor_at(n - mm, idx, &mut tb);
                    let w = -(mm as f64) / nf;
                    if d == 2 {
                        // ε_{zxy} = +1, ε_{zyx} = -1 : Σ_l (A_{lx} B_{ly} - A_{ly} B_{lx})
                        let mut v = 0.0;
                        for l in 0..2 {
                            v += ta[l * 2] * tb[l * 2 + 1] - ta[l * 2 + 1] * tb[l * 2];
                        }
                        t[0][idx] += w * v;
                    } else {
                        for i in 0..3 {
                            let j = (i + 1) % 3;
                            let k = (i + 2) % 3;
                            let mut v = 0.0;
                            for l in 0..3 {
                                v += ta[l * 3 + j] * tb[l * 3 + k] - ta[l * 3 + k] * tb[l * 3 + j];
                            }
                            t[i][idx] += w * v;
                        }
                    }
                }
            }
        }
        (s, t)
    }

    /// Eulerian positions x = q + Σ D^n Ψ^(n), interleaved [idx*dim + a], not wrapped.
    pub fn positions(&self, dgrow: f64, order: usize, out: &mut [f32]) {
        let d = self.grid.dim;
        let dx = self.grid.dx();
        let order = order.min(self.order).max(1);
        for idx in 0..self.grid.size {
            let ijk = self.grid.unravel(idx);
            for a in 0..d {
                let mut x = ijk[a] as f64 * dx;
                let mut dn = 1.0;
                for n in 1..=order {
                    dn *= dgrow;
                    x += dn * self.psi[n - 1][a][idx] as f64;
                }
                out[idx * d + a] = x as f32;
            }
        }
    }

    /// Velocities v = dx/dD = Σ n D^{n-1} Ψ^(n), interleaved.
    pub fn velocities(&self, dgrow: f64, order: usize, out: &mut [f32]) {
        let d = self.grid.dim;
        let order = order.min(self.order).max(1);
        for idx in 0..self.grid.size {
            for a in 0..d {
                let mut v = 0.0;
                let mut dn = 1.0; // D^{n-1}
                for n in 1..=order {
                    v += n as f64 * dn * self.psi[n - 1][a][idx] as f64;
                    dn *= dgrow;
                }
                out[idx * d + a] = v as f32;
            }
        }
    }

    /// Jacobian J(q, D) = det(I + Σ D^n M^(n)).
    pub fn jacobian(&self, dgrow: f64, order: usize, out: &mut [f64]) {
        let d = self.grid.dim;
        let order = order.min(self.order).max(1);
        let mut mt = vec![0.0; d * d];
        for idx in 0..self.grid.size {
            for c in 0..d * d {
                let mut v = 0.0;
                let mut dn = 1.0;
                for n in 1..=order {
                    dn *= dgrow;
                    v += dn * self.m[n - 1][c][idx] as f64;
                }
                mt[c] = v;
            }
            out[idx] = det_i_plus(&mt, d);
        }
    }

    /// Minimum Jacobian over the grid at time D.
    pub fn min_jacobian(&self, dgrow: f64, order: usize) -> f64 {
        let mut j = vec![0.0; self.grid.size];
        self.jacobian(dgrow, order, &mut j);
        j.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// First shell-crossing time: smallest D > 0 with min_q J(q,D) = 0.
    pub fn shell_crossing(&self, order: usize) -> f64 {
        // bracket on a log grid, then bisect
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        let mut dd = 1e-3;
        for _ in 0..80 {
            if self.min_jacobian(dd, order) <= 0.0 {
                hi = dd;
                break;
            }
            lo = dd;
            dd *= 1.25;
        }
        if hi == 0.0 {
            return f64::INFINITY;
        }
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.min_jacobian(mid, order) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }
}
