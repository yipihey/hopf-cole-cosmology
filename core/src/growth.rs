//! Time dependence of Lagrangian perturbation theory in a flat ΛCDM universe.
//!
//! Each *term* of the LPT series is a spatial field S^τ(q) times a time
//! function g_τ(t).  In Einstein–de Sitter g_τ = c_τ D^n with constant c_τ; in
//! ΛCDM every term obeys its own forced growth equation (Bouchet et al. 1995;
//! Rampf 2012; Matsubara 2015).  With ' = d/dln a and the operator
//! T̂g = H²[g'' + (2 + dlnH/dlna) g'], the longitudinal equation at order n is
//!
//!   T̂G - (3/2)Ω_m H² G = Σ_{(σ,τ)} [ (3/2)Ω_m H² g_σ g_τ - 2 g_σ T̂g_τ ] μ2(S^σ,S^τ)
//!                        + Σ_{(σ,ρ,τ)} [ (3/2)Ω_m H² g_σ g_ρ g_τ - 3 g_σ g_ρ T̂g_τ ] μ3(S^σ,S^ρ,S^τ),
//!
//! sums over *ordered* tuples of lower-order terms, and the transverse
//! (Cauchy-invariant) equation gives for a pair σ≠τ
//!
//!   ∇×Ψ ⊃ ε_{ijk} S^σ_{lj} S^τ_{lk} · h_{στ},   h_{στ} = -∫ (g_σ' g_τ - g_τ' g_σ) dln a .
//!
//! All ODEs are integrated together in ln a with RK4, starting deep in matter
//! domination from the EdS solution (growing modes only).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cosmology {
    /// Present-day matter density parameter; flat, Ω_Λ = 1 - Ω_m.
    pub omega_m: f64,
}

impl Cosmology {
    pub fn eds() -> Self { Cosmology { omega_m: 1.0 } }
    pub fn is_eds(&self) -> bool { (self.omega_m - 1.0).abs() < 1e-12 }
    /// E(a) = H/H0.
    pub fn e(&self, a: f64) -> f64 { (self.omega_m / (a * a * a) + (1.0 - self.omega_m)).sqrt() }
    /// Ω_m(a).
    pub fn omega_m_a(&self, a: f64) -> f64 { self.omega_m / (a * a * a) / self.e(a).powi(2) }
    /// dlnH/dlna = -(3/2) Ω_m(a) for flat ΛCDM.
    pub fn dlnh(&self, a: f64) -> f64 { -1.5 * self.omega_m_a(a) }
}

/// How a term is sourced by lower-order terms (indices into the term list).
#[derive(Clone, Debug, PartialEq)]
pub enum Parents {
    /// Ψ^(1) = -∇ϕ.
    Linear,
    /// Longitudinal, ∇·S = μ2(S^a, S^b)   (unordered pair, a ≤ b).
    Mu2(usize, usize),
    /// Longitudinal, ∇·S = μ3(S^a, S^b, S^c) (a ≤ b ≤ c; 3D only).
    Mu3(usize, usize, usize),
    /// Transverse, (∇×S)_i = ε_{ijk} S^a_{lj} S^b_{lk}  (a < b).
    Curl(usize, usize),
}

#[derive(Clone, Debug)]
pub struct TermSpec {
    pub order: usize,
    pub parents: Parents,
    pub label: String,
}

/// Enumerate the LPT terms up to `max_order` (μ3 terms only when dim == 3).
pub fn term_specs(max_order: usize, dim: usize) -> Vec<TermSpec> {
    let mut specs = vec![TermSpec { order: 1, parents: Parents::Linear, label: "1".into() }];
    for n in 2..=max_order {
        let existing: Vec<(usize, usize)> = specs.iter().enumerate().map(|(i, s)| (i, s.order)).collect();
        let mut new = Vec::new();
        // μ2 pairs
        for &(a, oa) in &existing {
            for &(b, ob) in &existing {
                if a <= b && oa + ob == n {
                    new.push(TermSpec { order: n, parents: Parents::Mu2(a, b), label: format!("{n}:μ2({},{})", specs[a].label, specs[b].label) });
                }
            }
        }
        // μ3 triples
        if dim == 3 {
            for &(a, oa) in &existing {
                for &(b, ob) in &existing {
                    for &(c, oc) in &existing {
                        if a <= b && b <= c && oa + ob + oc == n {
                            new.push(TermSpec { order: n, parents: Parents::Mu3(a, b, c), label: format!("{n}:μ3({},{},{})", specs[a].label, specs[b].label, specs[c].label) });
                        }
                    }
                }
            }
        }
        // transverse pairs
        for &(a, oa) in &existing {
            for &(b, ob) in &existing {
                if a < b && oa + ob == n {
                    new.push(TermSpec { order: n, parents: Parents::Curl(a, b), label: format!("{n}:T({},{})", specs[a].label, specs[b].label) });
                }
            }
        }
        specs.extend(new);
    }
    specs
}

/// Time functions of all terms.
#[derive(Clone, Debug)]
pub enum Growth {
    /// g_τ = coef[τ] D^order.
    Eds { coef: Vec<f64>, orders: Vec<usize> },
    /// Tables in ln a: g[τ][i], g'[τ][i]; d1 = g[0] is the linear growth factor
    /// normalized to a at early times.
    Lcdm { cosmo: Cosmology, lna: Vec<f64>, g: Vec<Vec<f64>>, gp: Vec<Vec<f64>>, gpp: Vec<Vec<f64>> },
}

/// Number of distinct orderings of a multiset given as a sorted tuple.
fn orderings3(a: usize, b: usize, c: usize) -> f64 {
    if a == b && b == c { 1.0 } else if a == b || b == c { 3.0 } else { 6.0 }
}

impl Growth {
    /// Einstein–de Sitter: closed-form coefficients from the same recursion.
    pub fn new_eds(specs: &[TermSpec]) -> Self {
        let mut coef: Vec<f64> = Vec::with_capacity(specs.len());
        let orders: Vec<usize> = specs.iter().map(|s| s.order).collect();
        // R(D^k) = k(k+1/2) D^{k-2}; T̂ ↔ R, (3/2)Ω_m H² ↔ 3/(2D²)
        let r = |k: usize| k as f64 * (k as f64 + 0.5);
        for s in specs {
            let n = s.order as f64;
            let c = match s.parents {
                Parents::Linear => 1.0,
                Parents::Mu2(a, b) => {
                    let (ca, cb, oa, ob) = (coef[a], coef[b], orders[a], orders[b]);
                    // ordered-pair sum: (3/2) g g - 2 g_σ R g_τ   for (σ,τ) = (a,b) and (b,a)
                    let f = if a == b { 1.5 * ca * cb - 2.0 * ca * cb * r(ob) } else { 2.0 * 1.5 * ca * cb - 2.0 * ca * cb * (r(ob) + r(oa)) };
                    f / ((n - 1.0) * (n + 1.5))
                }
                Parents::Mu3(a, b, c) => {
                    let (ca, cb, cc) = (coef[a], coef[b], coef[c]);
                    let (oa, ob, oc) = (orders[a], orders[b], orders[c]);
                    // ordered-triple sum of (3/2) ggg - 3 g g R g_last
                    let prod = ca * cb * cc;
                    let m = orderings3(a, b, c);
                    // Σ over orderings of R(order of last) = (m/3) Σ_i R(o_i)  (each element is last in m/3 orderings)
                    let f = m * 1.5 * prod - 3.0 * prod * (m / 3.0) * (r(oa) + r(ob) + r(oc));
                    f / ((n - 1.0) * (n + 1.5))
                }
                Parents::Curl(a, b) => {
                    let (ca, cb, oa, ob) = (coef[a], coef[b], orders[a] as f64, orders[b] as f64);
                    // h = -∫ (g_a' g_b - g_b' g_a) dD = (ob - oa)/n c_a c_b D^n
                    (ob - oa) / n * ca * cb
                }
            };
            coef.push(c);
        }
        Growth::Eds { coef, orders }
    }

    /// ΛCDM tables by RK4 in ln a from a = 1e-3 to a = a_max.
    pub fn new_lcdm(cosmo: Cosmology, specs: &[TermSpec], a_max: f64, npts: usize) -> Self {
        let eds = Growth::new_eds(specs);
        let (coef, orders) = match &eds { Growth::Eds { coef, orders } => (coef.clone(), orders.clone()), _ => unreachable!() };
        let nt = specs.len();
        let a0 = 1e-3f64;
        let lna0 = a0.ln();
        let lna1 = a_max.ln();
        let h = (lna1 - lna0) / (npts - 1) as f64;
        // state y = [g_0, g_0', g_1, g_1', ...]
        let mut y = vec![0.0; 2 * nt];
        for t in 0..nt {
            let n = orders[t] as f64;
            y[2 * t] = coef[t] * a0.powf(n);
            y[2 * t + 1] = n * coef[t] * a0.powf(n);
        }
        let rhs = |lna: f64, y: &[f64]| -> Vec<f64> {
            let a = lna.exp();
            let om = cosmo.omega_m_a(a);
            let damp = 2.0 + cosmo.dlnh(a);
            let mut dy = vec![0.0; 2 * nt];
            // g'' of each term in order (needed by children): tg[τ] = T̂g_τ / H² = g'' + damp g'
            let mut tg = vec![0.0; nt];
            for t in 0..nt {
                let g = |i: usize| y[2 * i];
                let gp = |i: usize| y[2 * i + 1];
                let gpp = match specs[t].parents {
                    Parents::Linear => -damp * gp(t) + 1.5 * om * g(t),
                    Parents::Mu2(a, b) => {
                        let f = if a == b { 1.5 * om * g(a) * g(b) - 2.0 * g(a) * tg[b] } else { 2.0 * 1.5 * om * g(a) * g(b) - 2.0 * (g(a) * tg[b] + g(b) * tg[a]) };
                        -damp * gp(t) + 1.5 * om * g(t) + f
                    }
                    Parents::Mu3(a, b, c) => {
                        let m = orderings3(a, b, c);
                        let prod = g(a) * g(b) * g(c);
                        // Σ over orderings of g g T̂g_last: each element last in m/3 orderings
                        let s = (m / 3.0) * (g(b) * g(c) * tg[a] + g(a) * g(c) * tg[b] + g(a) * g(b) * tg[c]);
                        let f = m * 1.5 * om * prod - 3.0 * s;
                        -damp * gp(t) + 1.5 * om * g(t) + f
                    }
                    Parents::Curl(a, b) => {
                        // h' = -(g_a' g_b - g_b' g_a); store h in g slot, h' in gp slot, "g''" = d/dlna of h'
                        // we integrate h directly: dy[2t] = h', and keep dy[2t+1] = d(h')/dlna
                        let hp = -(gp(a) * g(b) - gp(b) * g(a));
                        // d(h')/dlna = -(g_a'' g_b - g_b'' g_a)  (the g' g' terms cancel)
                        let gapp = tg[a] - damp * gp(a);
                        let gbpp = tg[b] - damp * gp(b);
                        let hpp = -(gapp * g(b) - gbpp * g(a));
                        dy[2 * t] = hp;
                        dy[2 * t + 1] = hpp;
                        tg[t] = hpp + damp * hp; // T̂h / H² for use as a parent
                        continue;
                    }
                };
                tg[t] = gpp + damp * gp(t);
                dy[2 * t] = gp(t);
                dy[2 * t + 1] = gpp;
            }
            dy
        };
        // for Curl terms the state slot y[2t+1] must hold h' consistently: initialize from the EdS form
        for t in 0..nt {
            if let Parents::Curl(_, _) = specs[t].parents {
                let n = orders[t] as f64;
                y[2 * t + 1] = n * coef[t] * a0.powf(n);
            }
        }
        let mut lna = vec![0.0; npts];
        let mut g = vec![vec![0.0; npts]; nt];
        let mut gp = vec![vec![0.0; npts]; nt];
        let mut gpp = vec![vec![0.0; npts]; nt];
        let mut x = lna0;
        for i in 0..npts {
            lna[i] = x;
            let k1 = rhs(x, &y);
            for t in 0..nt {
                g[t][i] = y[2 * t];
                gp[t][i] = y[2 * t + 1];
                gpp[t][i] = k1[2 * t + 1];
            }
            if i + 1 == npts { break; }
            let y2: Vec<f64> = y.iter().zip(&k1).map(|(v, k)| v + 0.5 * h * k).collect();
            let k2 = rhs(x + 0.5 * h, &y2);
            let y3: Vec<f64> = y.iter().zip(&k2).map(|(v, k)| v + 0.5 * h * k).collect();
            let k3 = rhs(x + 0.5 * h, &y3);
            let y4: Vec<f64> = y.iter().zip(&k3).map(|(v, k)| v + h * k).collect();
            let k4 = rhs(x + h, &y4);
            for j in 0..2 * nt {
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
            x += h;
        }
        Growth::Lcdm { cosmo, lna, g, gp, gpp }
    }

    pub fn is_eds(&self) -> bool { matches!(self, Growth::Eds { .. }) }
    pub fn nterms(&self) -> usize { match self { Growth::Eds { coef, .. } => coef.len(), Growth::Lcdm { g, .. } => g.len() } }

    /// Largest reachable linear growth factor (∞ for EdS).
    pub fn d_max(&self) -> f64 {
        match self {
            Growth::Eds { .. } => f64::INFINITY,
            Growth::Lcdm { g, .. } => *g[0].last().unwrap(),
        }
    }

    /// Linear growth factor at scale factor a (EdS: a).
    pub fn d_of_a(&self, a: f64) -> f64 {
        match self {
            Growth::Eds { .. } => a,
            Growth::Lcdm { lna, g, .. } => interp(lna, &g[0], a.ln()),
        }
    }

    /// Scale factor at which the linear growth factor equals d (inverse table).
    pub fn a_of_d(&self, d: f64) -> f64 {
        match self {
            Growth::Eds { .. } => d,
            Growth::Lcdm { lna, g, .. } => {
                let d1 = &g[0];
                let d = d.min(d1[d1.len() - 1]).max(d1[0]);
                // d1 is monotonic: bisection on the table
                let (mut lo, mut hi) = (0usize, d1.len() - 1);
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if d1[mid] <= d { lo = mid; } else { hi = mid; }
                }
                let f = if d1[hi] > d1[lo] { (d - d1[lo]) / (d1[hi] - d1[lo]) } else { 0.0 };
                (lna[lo] + f * (lna[hi] - lna[lo])).exp()
            }
        }
    }

    /// (g_τ, dg_τ/dD) at linear growth factor d.
    pub fn g_and_dg(&self, term: usize, d: f64) -> (f64, f64) {
        match self {
            Growth::Eds { coef, orders } => {
                let n = orders[term] as f64;
                (coef[term] * d.powf(n), n * coef[term] * d.powf(n - 1.0))
            }
            Growth::Lcdm { lna, g, gp, .. } => {
                let a = self.a_of_d(d);
                let x = a.ln();
                let gt = interp(lna, &g[term], x);
                let gpt = interp(lna, &gp[term], x);
                let d1p = interp(lna, &gp[0], x);
                (gt, if d1p != 0.0 { gpt / d1p } else { 0.0 })
            }
        }
    }

    /// Ω_m(a) at the given D (1 for EdS).
    pub fn omega_m_at_d(&self, d: f64) -> f64 {
        match self {
            Growth::Eds { .. } => 1.0,
            Growth::Lcdm { cosmo, .. } => cosmo.omega_m_a(self.a_of_d(d)),
        }
    }
}

/// Linear interpolation on a uniform table (clamped).
fn interp(x: &[f64], y: &[f64], xq: f64) -> f64 {
    let n = x.len();
    if xq <= x[0] { return y[0]; }
    if xq >= x[n - 1] { return y[n - 1]; }
    let h = (x[n - 1] - x[0]) / (n - 1) as f64;
    let i = ((xq - x[0]) / h).floor() as usize;
    let i = i.min(n - 2);
    let f = (xq - x[i]) / h;
    y[i] * (1.0 - f) + y[i + 1] * f
}

/// Growth-rate f = dlnD/dlna at scale factor a.
pub fn growth_rate(growth: &Growth, a: f64) -> f64 {
    match growth {
        Growth::Eds { .. } => 1.0,
        Growth::Lcdm { lna, g, gp, .. } => interp(lna, &gp[0], a.ln()) / interp(lna, &g[0], a.ln()),
    }
}

#[allow(dead_code)]
fn _unused() -> f64 { PI }
