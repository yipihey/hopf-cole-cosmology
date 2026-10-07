//! Lagrangian perturbation theory to arbitrary order, in 2D (= 3D with
//! translational symmetry along z) and 3D, with Einstein–de Sitter (closed
//! form D^n) or exact ΛCDM time dependence.
//!
//! Conventions
//! -----------
//! x(q, D) = q + Σ_τ g_τ(D) S^τ(q),   δ_lin(D) = D δ0,   S^1 = -∇ϕ,  ∇²ϕ = δ0.
//!
//! Each term τ has a spatial field S^τ sourced by lower-order terms (see
//! `growth::term_specs`) and a time function g_τ (see `growth::Growth`).  In
//! EdS the terms of equal order are proportional to D^n and are merged into a
//! single field per order, reproducing the Rampf & Hahn (2021) recursion
//!
//!   ∇·Ψ^(n) = Σ_{a+b=n} c2(n;a,b) μ2(M^a,M^b) + Σ_{a+b+c=n} c3(n;a,b,c) μ3(M^a,M^b,M^c),
//!   (∇×Ψ^(n))_i = -(1/n) Σ_{m} m ε_{ijk} M^(m)_{lj} M^(n-m)_{lk},
//!
//! which is validated by the EOM-residual tests in tests/validate.rs.

use crate::fft::FftEngine;
use crate::grid::Grid;
use crate::growth::{term_specs, Cosmology, Growth, Parents, TermSpec};
use num_complex::Complex64 as C64;

pub struct Term {
    pub spec: TermSpec,
    /// psi[a]: a-th component of S^τ on the Lagrangian grid.
    pub psi: Vec<Vec<f32>>,
    /// m[a*dim+b] = ∂_b S^τ_a (kept while building; dropped afterwards in 3D).
    pub m: Option<Vec<Vec<f32>>>,
    /// Longitudinal source ∇·S (empty for transverse terms).
    pub div: Vec<f32>,
    /// Transverse source (1 component in 2D, 3 in 3D; empty for longitudinal terms).
    pub curl: Vec<Vec<f32>>,
}

pub struct Lpt {
    pub grid: Grid,
    pub order: usize,
    pub terms: Vec<Term>,
    pub growth: Growth,
    pub cosmo: Cosmology,
}

#[inline]
fn tr(a: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d { s += a[i * d + i]; }
    s
}
#[inline]
fn tr_ab(a: &[f64], b: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d { for j in 0..d { s += a[i * d + j] * b[j * d + i]; } }
    s
}
#[inline]
fn tr_abc(a: &[f64], b: &[f64], c: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d { for j in 0..d { for k in 0..d { s += a[i * d + j] * b[j * d + k] * c[k * d + i]; } } }
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
/// det(I + M) for d = 1, 2 or 3.
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

/// 4th-order central derivative ∂_axis of a periodic f32 field, as f64.
fn fd_gradient(grid: &Grid, f: &[f32], axis: usize) -> Vec<f64> {
    let n = grid.n;
    let d = grid.dim;
    let stride = n.pow((d - 1 - axis) as u32);
    let block = stride * n;
    let h = grid.dx();
    let mut out = vec![0.0; grid.size];
    for idx in 0..grid.size {
        let b = idx / block;
        let within = idx % block;
        let i = (within / stride) as i64;
        let s0 = within % stride;
        let at = |j: i64| -> f64 { f[b * block + (j.rem_euclid(n as i64) as usize) * stride + s0] as f64 };
        out[idx] = (8.0 * (at(i + 1) - at(i - 1)) - (at(i + 2) - at(i - 2))) / (12.0 * h);
    }
    out
}

impl Lpt {
    /// Build the LPT terms from the Fourier-space linear density δ0_hat.
    /// `keep_tensors`: keep the gradient tensors of all terms (needed by the
    /// tests and by exact Jacobians; costs dim² fields per term).
    pub fn new(grid: &Grid, eng: &mut FftEngine, delta0_hat: &[C64], order: usize, cosmo: Cosmology, keep_tensors: bool) -> Self {
        let d = grid.dim;
        let specs = term_specs(order, d);
        let growth = if cosmo.is_eds() { Growth::new_eds(&specs) } else { Growth::new_lcdm(cosmo, &specs, 20.0, 4000) };
        let mut lpt = Lpt { grid: grid.clone(), order, terms: Vec::new(), growth, cosmo };

        for spec in specs {
            let (psihat, div, curl): (Vec<Vec<C64>>, Vec<f32>, Vec<Vec<f32>>) = match spec.parents {
                Parents::Linear => {
                    let mut psihat = Vec::new();
                    for a in 0..d {
                        let v: Vec<C64> = (0..grid.size)
                            .map(|idx| {
                                let k2 = grid.k2(idx);
                                if k2 == 0.0 { C64::new(0.0, 0.0) } else {
                                    let k = grid.kvec(idx);
                                    C64::new(0.0, k[a] / k2) * delta0_hat[idx]
                                }
                            })
                            .collect();
                        psihat.push(v);
                    }
                    let div = eng.inverse_to_real_f32(delta0_hat.iter().map(|v| -v).collect());
                    (psihat, div, vec![])
                }
                Parents::Mu2(a, b) => {
                    let s = lpt.source_mu2(a, b);
                    let shat = eng.forward_real(&s);
                    (lpt.solve_longitudinal(&shat), s.iter().map(|&v| v as f32).collect(), vec![])
                }
                Parents::Mu3(a, b, c) => {
                    let s = lpt.source_mu3(a, b, c);
                    let shat = eng.forward_real(&s);
                    (lpt.solve_longitudinal(&shat), s.iter().map(|&v| v as f32).collect(), vec![])
                }
                Parents::Curl(a, b) => {
                    let t = lpt.source_curl(a, b);
                    let that: Vec<Vec<C64>> = t.iter().map(|tc| eng.forward_real(tc)).collect();
                    (lpt.solve_transverse(&that), vec![], t.iter().map(|tc| tc.iter().map(|&v| v as f32).collect()).collect())
                }
            };
            let mut psi = Vec::new();
            let mut m = vec![Vec::new(); d * d];
            for a in 0..d {
                psi.push(eng.inverse_to_real_f32(psihat[a].clone()));
                for b in 0..d {
                    let dh: Vec<C64> = psihat[a].iter().enumerate().map(|(idx, &v)| { let k = grid.kvec(idx); C64::new(0.0, k[b]) * v }).collect();
                    m[a * d + b] = eng.inverse_to_real_f32(dh);
                }
            }
            lpt.terms.push(Term { spec, psi, m: Some(m), div, curl });
        }
        if lpt.growth.is_eds() {
            lpt.merge_eds_orders();
        }
        if !keep_tensors {
            for t in lpt.terms.iter_mut() { t.m = None; }
        }
        lpt
    }

    /// Solve ∇·S = Ŝ for a potential field: Ŝ_a = -i k_a Ŝ / k².
    fn solve_longitudinal(&self, shat: &[C64]) -> Vec<Vec<C64>> {
        let grid = &self.grid;
        let d = grid.dim;
        let mut out = vec![vec![C64::new(0.0, 0.0); grid.size]; d];
        for idx in 0..grid.size {
            let k2 = grid.k2(idx);
            if k2 == 0.0 { continue; }
            let k = grid.kvec(idx);
            for a in 0..d { out[a][idx] = C64::new(0.0, -k[a] / k2) * shat[idx]; }
        }
        out
    }
    /// Solve ∇×S = T for a divergence-free field: Ŝ = i (k × T̂)/k².
    fn solve_transverse(&self, that: &[Vec<C64>]) -> Vec<Vec<C64>> {
        let grid = &self.grid;
        let d = grid.dim;
        let mut out = vec![vec![C64::new(0.0, 0.0); grid.size]; d];
        for idx in 0..grid.size {
            let k2 = grid.k2(idx);
            if k2 == 0.0 { continue; }
            let k = grid.kvec(idx);
            if d == 2 {
                let tz = that[0][idx];
                out[0][idx] = C64::new(0.0, k[1] / k2) * tz;
                out[1][idx] = C64::new(0.0, -k[0] / k2) * tz;
            } else {
                let (tx, ty, tz) = (that[0][idx], that[1][idx], that[2][idx]);
                out[0][idx] = C64::new(0.0, 1.0 / k2) * (k[1] * tz - k[2] * ty);
                out[1][idx] = C64::new(0.0, 1.0 / k2) * (k[2] * tx - k[0] * tz);
                out[2][idx] = C64::new(0.0, 1.0 / k2) * (k[0] * ty - k[1] * tx);
            }
        }
        out
    }

    #[inline]
    fn tensor_at(&self, t: usize, idx: usize, out: &mut [f64]) {
        let d = self.grid.dim;
        let m = self.terms[t].m.as_ref().expect("gradient tensors were dropped");
        for c in 0..d * d { out[c] = m[c][idx] as f64; }
    }
    fn source_mu2(&self, a: usize, b: usize) -> Vec<f64> {
        let d = self.grid.dim;
        let (mut ta, mut tb) = (vec![0.0; 9], vec![0.0; 9]);
        (0..self.grid.size).map(|idx| { self.tensor_at(a, idx, &mut ta); self.tensor_at(b, idx, &mut tb); mu2(&ta[..d * d], &tb[..d * d], d) }).collect()
    }
    fn source_mu3(&self, a: usize, b: usize, c: usize) -> Vec<f64> {
        let (mut ta, mut tb, mut tc) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        (0..self.grid.size).map(|idx| { self.tensor_at(a, idx, &mut ta); self.tensor_at(b, idx, &mut tb); self.tensor_at(c, idx, &mut tc); mu3(&ta, &tb, &tc, 3) }).collect()
    }
    /// T_i = ε_{ijk} S^a_{lj} S^b_{lk}
    fn source_curl(&self, a: usize, b: usize) -> Vec<Vec<f64>> {
        let d = self.grid.dim;
        let size = self.grid.size;
        let ncurl = if d == 2 { 1 } else { 3 };
        let mut t = vec![vec![0.0; size]; ncurl];
        let (mut ta, mut tb) = (vec![0.0; 9], vec![0.0; 9]);
        for idx in 0..size {
            self.tensor_at(a, idx, &mut ta);
            self.tensor_at(b, idx, &mut tb);
            if d == 2 {
                let mut v = 0.0;
                for l in 0..2 { v += ta[l * 2] * tb[l * 2 + 1] - ta[l * 2 + 1] * tb[l * 2]; }
                t[0][idx] = v;
            } else {
                for i in 0..3 {
                    let j = (i + 1) % 3;
                    let k = (i + 2) % 3;
                    let mut v = 0.0;
                    for l in 0..3 { v += ta[l * 3 + j] * tb[l * 3 + k] - ta[l * 3 + k] * tb[l * 3 + j]; }
                    t[i][idx] = v;
                }
            }
        }
        t
    }

    /// In EdS all terms of one order are ∝ D^n: merge them into one term per
    /// order with g = D^n (the classical Ψ^(n)).
    fn merge_eds_orders(&mut self) {
        let coef: Vec<f64> = match &self.growth { Growth::Eds { coef, .. } => coef.clone(), _ => return };
        let d = self.grid.dim;
        let size = self.grid.size;
        let mut merged: Vec<Term> = Vec::new();
        let mut new_coef = Vec::new();
        let mut new_orders = Vec::new();
        for n in 1..=self.order {
            let idxs: Vec<usize> = (0..self.terms.len()).filter(|&i| self.terms[i].spec.order == n).collect();
            if idxs.is_empty() { continue; }
            let ncurl = if d == 2 { 1 } else { 3 };
            let mut psi = vec![vec![0.0f32; size]; d];
            let mut m = vec![vec![0.0f32; size]; d * d];
            let mut div = vec![0.0f32; size];
            let mut curl = vec![vec![0.0f32; size]; ncurl];
            let mut has_curl = false;
            for &i in &idxs {
                let c = coef[i] as f32;
                let t = &self.terms[i];
                for a in 0..d { for idx in 0..size { psi[a][idx] += c * t.psi[a][idx]; } }
                if let Some(tm) = &t.m { for comp in 0..d * d { for idx in 0..size { m[comp][idx] += c * tm[comp][idx]; } } }
                if !t.div.is_empty() { for idx in 0..size { div[idx] += c * t.div[idx]; } }
                if !t.curl.is_empty() { has_curl = true; for comp in 0..ncurl { for idx in 0..size { curl[comp][idx] += c * t.curl[comp][idx]; } } }
            }
            merged.push(Term {
                spec: TermSpec { order: n, parents: if n == 1 { Parents::Linear } else { Parents::Mu2(0, 0) }, label: format!("{n}") },
                psi, m: Some(m), div, curl: if has_curl { curl } else { vec![] },
            });
            new_coef.push(1.0);
            new_orders.push(n);
        }
        self.terms = merged;
        self.growth = Growth::Eds { coef: new_coef, orders: new_orders };
    }

    /// Term indices with order ≤ `order`.
    fn active(&self, order: usize) -> Vec<usize> {
        (0..self.terms.len()).filter(|&i| self.terms[i].spec.order <= order).collect()
    }

    /// Eulerian positions x = q + Σ g_τ(D) S^τ, interleaved [idx*dim + a], not wrapped.
    pub fn positions(&self, dgrow: f64, order: usize, out: &mut [f32]) {
        let d = self.grid.dim;
        let dx = self.grid.dx();
        let order = order.min(self.order).max(1);
        let act = self.active(order);
        let g: Vec<f64> = act.iter().map(|&t| self.growth.g_and_dg(t, dgrow).0).collect();
        for idx in 0..self.grid.size {
            let ijk = self.grid.unravel(idx);
            for a in 0..d {
                let mut x = ijk[a] as f64 * dx;
                for (j, &t) in act.iter().enumerate() { x += g[j] * self.terms[t].psi[a][idx] as f64; }
                out[idx * d + a] = x as f32;
            }
        }
    }

    /// Velocities v = dx/dD = Σ (dg_τ/dD) S^τ, interleaved.
    pub fn velocities(&self, dgrow: f64, order: usize, out: &mut [f32]) {
        let d = self.grid.dim;
        let order = order.min(self.order).max(1);
        let act = self.active(order);
        let dg: Vec<f64> = act.iter().map(|&t| self.growth.g_and_dg(t, dgrow).1).collect();
        for idx in 0..self.grid.size {
            for a in 0..d {
                let mut v = 0.0;
                for (j, &t) in act.iter().enumerate() { v += dg[j] * self.terms[t].psi[a][idx] as f64; }
                out[idx * d + a] = v as f32;
            }
        }
    }

    /// Displacement Ψ(D) = Σ g_τ S^τ up to `order`, one component.
    pub fn displacement_component(&self, dgrow: f64, order: usize, a: usize) -> Vec<f32> {
        let act = self.active(order.min(self.order).max(1));
        let g: Vec<f64> = act.iter().map(|&t| self.growth.g_and_dg(t, dgrow).0).collect();
        (0..self.grid.size).map(|idx| { let mut v = 0.0; for (j, &t) in act.iter().enumerate() { v += g[j] * self.terms[t].psi[a][idx] as f64; } v as f32 }).collect()
    }

    /// Longitudinal source Σ_{τ∈order} g_τ(D) ∇·S^τ / D^order  (equals the EdS ∇·Ψ^(n) shape).
    pub fn div_source(&self, order: usize, dgrow: f64) -> Vec<f32> {
        let size = self.grid.size;
        let mut out = vec![0.0f32; size];
        let dn = dgrow.max(1e-12).powi(order as i32);
        for (t, term) in self.terms.iter().enumerate() {
            if term.spec.order != order || term.div.is_empty() { continue; }
            let w = (self.growth.g_and_dg(t, dgrow.max(1e-12)).0 / dn) as f32;
            for idx in 0..size { out[idx] += w * term.div[idx]; }
        }
        out
    }
    /// Transverse source component, same normalization as `div_source` (zeros if none).
    pub fn curl_source(&self, order: usize, comp: usize, dgrow: f64) -> Vec<f32> {
        let size = self.grid.size;
        let mut out = vec![0.0f32; size];
        let dn = dgrow.max(1e-12).powi(order as i32);
        for (t, term) in self.terms.iter().enumerate() {
            if term.spec.order != order || term.curl.is_empty() || comp >= term.curl.len() { continue; }
            let w = (self.growth.g_and_dg(t, dgrow.max(1e-12)).0 / dn) as f32;
            for idx in 0..size { out[idx] += w * term.curl[comp][idx]; }
        }
        out
    }

    /// Gradient tensor of the total displacement at D: M_{ab} = ∂_b Ψ_a.  Uses
    /// the spectral tensors when kept, otherwise 4th-order finite differences.
    pub fn displacement_gradient(&self, dgrow: f64, order: usize) -> Vec<Vec<f64>> {
        let d = self.grid.dim;
        let size = self.grid.size;
        let act = self.active(order.min(self.order).max(1));
        let g: Vec<f64> = act.iter().map(|&t| self.growth.g_and_dg(t, dgrow).0).collect();
        let mut out = vec![vec![0.0f64; size]; d * d];
        let have_tensors = act.iter().all(|&t| self.terms[t].m.is_some());
        if have_tensors {
            for (j, &t) in act.iter().enumerate() {
                let m = self.terms[t].m.as_ref().unwrap();
                for c in 0..d * d { for idx in 0..size { out[c][idx] += g[j] * m[c][idx] as f64; } }
            }
        } else {
            for a in 0..d {
                let psi_a = self.displacement_component(dgrow, order, a);
                for b in 0..d {
                    let gab = fd_gradient(&self.grid, &psi_a, b);
                    out[a * d + b] = gab;
                }
            }
        }
        out
    }

    /// Jacobian J(q, D) = det(I + ∇Ψ).
    pub fn jacobian(&self, dgrow: f64, order: usize, out: &mut [f64]) {
        let d = self.grid.dim;
        let m = self.displacement_gradient(dgrow, order);
        let mut mt = vec![0.0; d * d];
        for idx in 0..self.grid.size {
            for c in 0..d * d { mt[c] = m[c][idx]; }
            out[idx] = det_i_plus(&mt, d);
        }
    }

    pub fn min_jacobian(&self, dgrow: f64, order: usize) -> f64 {
        let mut j = vec![0.0; self.grid.size];
        self.jacobian(dgrow, order, &mut j);
        j.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Zel'dovich (1LPT) shell-crossing time 1/max_q(-λ_min(∇S^1)).
    pub fn zeldovich_shell_crossing(&self) -> f64 {
        let d = self.grid.dim;
        let m1 = self.displacement_gradient(1.0, 1);
        let mut lam_max = 0.0f64;
        let mut m = vec![0.0; 9];
        for idx in 0..self.grid.size {
            for c in 0..d * d { m[c] = m1[c][idx]; }
            let lmin = if d == 2 {
                let tr = m[0] + m[3];
                let det = m[0] * m[3] - m[1] * m[2];
                0.5 * tr - (0.25 * tr * tr - det).max(0.0).sqrt()
            } else if d == 3 {
                let a = [m[0], m[4], m[8], 0.5 * (m[1] + m[3]), 0.5 * (m[2] + m[6]), 0.5 * (m[5] + m[7])];
                let q = (a[0] + a[1] + a[2]) / 3.0;
                let p2 = (a[0] - q).powi(2) + (a[1] - q).powi(2) + (a[2] - q).powi(2) + 2.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]);
                let pp = (p2 / 6.0).sqrt();
                if pp < 1e-300 { q } else {
                    let b = [(a[0] - q) / pp, (a[1] - q) / pp, (a[2] - q) / pp, a[3] / pp, a[4] / pp, a[5] / pp];
                    let detb = b[0] * (b[1] * b[2] - b[5] * b[5]) - b[3] * (b[3] * b[2] - b[5] * b[4]) + b[4] * (b[3] * b[5] - b[1] * b[4]);
                    let r = (detb / 2.0).clamp(-1.0, 1.0);
                    let phi = r.acos() / 3.0;
                    q + 2.0 * pp * (phi + 2.0 * std::f64::consts::PI / 3.0).cos()
                }
            } else { m[0] };
            lam_max = lam_max.max(-lmin);
        }
        // In ΛCDM the 1LPT term carries g_1 = D exactly, so the formula is unchanged.
        if lam_max > 0.0 { 1.0 / lam_max } else { f64::INFINITY }
    }

    /// First shell-crossing time: smallest D > 0 with min_q J(q,D) = 0.
    pub fn shell_crossing(&self, order: usize) -> f64 {
        let dmax = self.growth.d_max();
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        let mut dd = 0.5 * self.zeldovich_shell_crossing().min(1e3).max(1e-3);
        for _ in 0..40 {
            if dd > dmax { return f64::INFINITY; }
            if self.min_jacobian(dd, order) <= 0.0 { hi = dd; break; }
            lo = dd;
            dd *= 1.6;
        }
        if hi == 0.0 { return f64::INFINITY; }
        for _ in 0..16 {
            let mid = 0.5 * (lo + hi);
            if self.min_jacobian(mid, order) <= 0.0 { hi = mid; } else { lo = mid; }
        }
        0.5 * (lo + hi)
    }

    /// Labels of all terms (for the UI).
    pub fn term_labels(&self) -> Vec<String> { self.terms.iter().map(|t| t.spec.label.clone()).collect() }
}
