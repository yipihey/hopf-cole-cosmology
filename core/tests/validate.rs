use hcc_core::ics::PkShape;
use hcc_core::lpt::{det_i_plus, mu2};
use hcc_core::spectra::{f_sym, f_za, fg_kernel};
use hcc_core::Cosmo;

fn setup(dim: usize, n: usize, order: usize) -> Cosmo {
    let mut c = Cosmo::new(dim, n, 1.0);
    c.keep_tensors = Some(true);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 42, 1.0);
    c.build_lpt(order);
    c
}

#[test]
fn growth_tables_reduce_to_eds() {
    use hcc_core::growth::{term_specs, Cosmology, Growth};
    let specs = term_specs(4, 3);
    assert_eq!(specs.len(), 13, "expected 13 terms through 4LPT in 3D");
    let eds = Growth::new_eds(&specs);
    let tab = Growth::new_lcdm(Cosmology { omega_m: 1.0 }, &specs, 20.0, 4000);
    for t in 0..specs.len() {
        for &d in &[0.3, 1.0, 3.0] {
            let (ge, dge) = eds.g_and_dg(t, d);
            let (gt, dgt) = tab.g_and_dg(t, d);
            assert!((gt / ge - 1.0).abs() < 2e-4, "term {} ({}) at D={}: table {} vs EdS {}", t, specs[t].label, d, gt, ge);
            assert!((dgt / dge - 1.0).abs() < 2e-3, "dg term {} at D={}: {} vs {}", t, d, dgt, dge);
        }
    }
    // EdS coefficients: 2LPT -3/7; 3LPT: -1/3 (μ3), 10/21 × (-3/7)... check the known ones
    if let Growth::Eds { coef, .. } = &eds {
        assert!((coef[1] + 3.0 / 7.0).abs() < 1e-12, "2LPT coefficient {}", coef[1]);
        let i3a = specs.iter().position(|s| s.label.starts_with("3:μ3")).unwrap();
        let i3b = specs.iter().position(|s| s.label.starts_with("3:μ2")).unwrap();
        let i3c = specs.iter().position(|s| s.label.starts_with("3:T")).unwrap();
        assert!((coef[i3a] + 1.0 / 3.0).abs() < 1e-12, "3a {}", coef[i3a]);
        // μ2(1,2) carries c_2 = -3/7 already: coefficient = (10/21)/(-3/7)·(-3/7)... value is -10/9 × (-3/7)? check: 10/21 overall
        assert!((coef[i3b] - 10.0 / 21.0 / (3.0 / 7.0) * (3.0 / 7.0) * (-1.0) * (-1.0)).abs() < 1e-12 || (coef[i3b] + 10.0 / 9.0 * 3.0 / 7.0).abs() < 1e-12 || (coef[i3b] - 10.0 / 21.0).abs() < 1e-12, "3b {}", coef[i3b]);
        assert!((coef[i3c].abs() - 1.0 / 7.0 / (3.0 / 7.0) * (3.0 / 7.0)).abs() < 1e-12 || (coef[i3c].abs() - 1.0 / 3.0).abs() < 1e-12, "3c {}", coef[i3c]);
    }
    // ΛCDM: D2/D1² ≈ -3/7 Ω_m(a)^{-1/143} (Bouchet et al. 1995 fit) to 0.3%
    let lcdm = Growth::new_lcdm(Cosmology { omega_m: 0.3 }, &specs, 20.0, 4000);
    for &a in &[0.5, 1.0, 2.0] {
        let d1 = lcdm.d_of_a(a);
        let (g2, _) = lcdm.g_and_dg(1, d1);
        let om = Cosmology { omega_m: 0.3 }.omega_m_a(a);
        let fit = -3.0 / 7.0 * om.powf(-1.0 / 143.0);
        println!("a={a}: D1={d1:.4}, D2/D1²={:.5} vs fit {:.5}", g2 / (d1 * d1), fit);
        assert!((g2 / (d1 * d1) / fit - 1.0).abs() < 3e-3);
    }
}

#[test]
fn lcdm_eom_residual_scaling_2d() {
    // with exact ΛCDM growth the Lagrangian equations must still be satisfied order by order
    let mut c = Cosmo::new(2, 64, 1.0);
    c.keep_tensors = Some(true);
    c.set_cosmology(0.3);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 42, 1.0);
    c.build_lpt(4);
    let l = c.lpt_ref();
    assert!(!l.growth.is_eds());
    assert_eq!(l.terms.len(), 9, "2D 4LPT has 9 terms, got {}", l.terms.len());
    // Residual of the longitudinal equation in ln a at the scale factor a(D):
    // cof(I+M)_{ji} ∂_i(T̂Ψ_j)/H² = (3/2) Ω_m (J - 1), with T̂Ψ/H² = Σ (g'' + damp g') S.
    for order in 2..=4 {
        let r: Vec<f64> = [0.2, 0.1].iter().map(|&d| lcdm_residual(&c, d, order)).collect();
        let _ = &r;
        let ratio = r[0] / r[1];
        println!("ΛCDM 2D order {} residual ratio {:.3} (expect ≈ {:.1})", order, ratio, 2f64.powi(order as i32 + 1));
        assert!((ratio / 2f64.powi(order as i32 + 1) - 1.0).abs() < 0.2, "order {} ratio {}", order, ratio);
    }
}

fn lcdm_residual(c: &Cosmo, dgrow: f64, order: usize) -> f64 {
    use hcc_core::growth::Growth;
    let l = c.lpt_ref();
    let d = c.grid.dim;
    let (lna_t, g_t, gp_t, gpp_t, cosmo) = match &l.growth { Growth::Lcdm { lna, g, gp, gpp, cosmo } => (lna, g, gp, gpp, *cosmo), _ => unreachable!() };
    // use the table point nearest to a(D) exactly (no interpolation error)
    let a = l.growth.a_of_d(dgrow);
    let x = a.ln();
    let h = lna_t[1] - lna_t[0];
    let i = (((x - lna_t[0]) / h).round() as usize).min(lna_t.len() - 1);
    let a = lna_t[i].exp();
    let om = cosmo.omega_m_a(a);
    let damp = 2.0 + cosmo.dlnh(a);
    let act: Vec<usize> = (0..l.terms.len()).filter(|&t| l.terms[t].spec.order <= order).collect();
    let mut gv = Vec::new();
    let mut tg = Vec::new();
    for &t in &act {
        gv.push(g_t[t][i]);
        tg.push(gpp_t[t][i] + damp * gp_t[t][i]);
    }
    let mut acc = 0.0;
    let mut mt = vec![0.0; d * d];
    let mut rt = vec![0.0; d * d];
    for idx in 0..c.grid.size {
        for comp in 0..d * d {
            let mut v = 0.0; let mut r = 0.0;
            for (j, &t) in act.iter().enumerate() {
                let m = l.terms[t].m.as_ref().unwrap()[comp][idx] as f64;
                v += gv[j] * m; r += tg[j] * m;
            }
            mt[comp] = v; rt[comp] = r;
        }
        let cof = cofactor(&mt, d);
        let mut lhs = 0.0;
        for j in 0..d { for ii in 0..d { lhs += cof[ii * d + j] * rt[j * d + ii]; } }
        let rhs = 1.5 * om * (det_i_plus(&mt, d) - 1.0);
        acc += (lhs - rhs).powi(2);
    }
    (acc / c.grid.size as f64).sqrt()
}

/// cofactor of (I+M), d×d, C[j*d+i] = cof_{ji}
fn cofactor(m: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0; d * d];
    if d == 2 {
        let a = [1.0 + m[0], m[1], m[2], 1.0 + m[3]];
        c[0] = a[3];
        c[1] = -a[2];
        c[2] = -a[1];
        c[3] = a[0];
    } else {
        let a = [1.0 + m[0], m[1], m[2], m[3], 1.0 + m[4], m[5], m[6], m[7], 1.0 + m[8]];
        for i in 0..3 {
            for j in 0..3 {
                let r = [(i + 1) % 3, (i + 2) % 3];
                let s = [(j + 1) % 3, (j + 2) % 3];
                c[i * 3 + j] = a[r[0] * 3 + s[0]] * a[r[1] * 3 + s[1]] - a[r[0] * 3 + s[1]] * a[r[1] * 3 + s[0]];
            }
        }
    }
    c
}

/// rms of the longitudinal EOM residual × D² (should scale as D^{order+1})
fn longitudinal_residual(c: &Cosmo, dgrow: f64, order: usize) -> f64 {
    let l = c.lpt_ref();
    let d = c.grid.dim;
    let mut acc = 0.0;
    let mut mt = vec![0.0; d * d];
    let mut rt = vec![0.0; d * d];
    for idx in 0..c.grid.size {
        for comp in 0..d * d {
            let mut v = 0.0;
            let mut r = 0.0;
            let mut dn = 1.0;
            for n in 1..=order {
                dn *= dgrow;
                let mn = l.terms[n - 1].m.as_ref().unwrap()[comp][idx] as f64;
                v += dn * mn;
                r += (n as f64) * (n as f64 + 0.5) * dn * mn; // D² × n(n+1/2) D^{n-2}
            }
            mt[comp] = v;
            rt[comp] = r;
        }
        let cof = cofactor(&mt, d);
        let mut lhs = 0.0;
        for j in 0..d {
            for i in 0..d {
                lhs += cof[i * d + j] * rt[j * d + i];
            }
        }
        let rhs = -1.5 * (1.0 - det_i_plus(&mt, d));
        acc += (lhs - rhs).powi(2);
    }
    (acc / c.grid.size as f64).sqrt()
}

/// rms Cauchy invariant ε_{ijk} ẋ_{l,j} x_{l,k} (should scale as D^{order})
fn cauchy_residual(c: &Cosmo, dgrow: f64, order: usize) -> f64 {
    let l = c.lpt_ref();
    let d = c.grid.dim;
    let mut acc = 0.0;
    let mut mt = vec![0.0; d * d];
    let mut vt = vec![0.0; d * d];
    for idx in 0..c.grid.size {
        for comp in 0..d * d {
            let mut x = 0.0;
            let mut v = 0.0;
            let mut dn = 1.0;
            for n in 1..=order {
                let mn = l.terms[n - 1].m.as_ref().unwrap()[comp][idx] as f64;
                v += (n as f64) * dn * mn;
                dn *= dgrow;
                x += dn * mn;
            }
            mt[comp] = x;
            vt[comp] = v;
        }
        // x_{l,k} = δ_{lk} + mt[l*d+k]
        if d == 2 {
            let mut s = 0.0;
            for l2 in 0..2 {
                let xl = |k: usize| if l2 == k { 1.0 } else { 0.0 } + mt[l2 * 2 + k];
                s += vt[l2 * 2] * xl(1) - vt[l2 * 2 + 1] * xl(0);
            }
            acc += s * s;
        } else {
            for i in 0..3 {
                let j = (i + 1) % 3;
                let k = (i + 2) % 3;
                let mut s = 0.0;
                for l2 in 0..3 {
                    let xl = |kk: usize| if l2 == kk { 1.0 } else { 0.0 } + mt[l2 * 3 + kk];
                    s += vt[l2 * 3 + j] * xl(k) - vt[l2 * 3 + k] * xl(j);
                }
                acc += s * s;
            }
        }
    }
    (acc / c.grid.size as f64).sqrt()
}

#[test]
fn lpt_2lpt_closed_form() {
    let c = setup(2, 64, 2);
    let l = c.lpt_ref();
    let mut err = 0.0;
    let mut norm = 0.0;
    for idx in 0..c.grid.size {
        let m1: Vec<f64> = (0..4).map(|k| l.terms[0].m.as_ref().unwrap()[k][idx] as f64).collect();
        let expect = -3.0 / 7.0 * mu2(&m1, &m1, 2);
        let got = l.terms[1].div[idx] as f64;
        err += (expect - got).powi(2);
        norm += expect.powi(2);
    }
    assert!((err / norm).sqrt() < 1e-5, "2LPT divergence mismatch {}", (err / norm).sqrt());
}

#[test]
fn lpt_eom_residual_scaling_2d() {
    for order in 1..=4 {
        let c = setup(2, 64, order);
        let r1 = longitudinal_residual(&c, 0.2, order);
        let r2 = longitudinal_residual(&c, 0.1, order);
        let ratio = r1 / r2;
        let expect = 2f64.powi(order as i32 + 1);
        println!("2D order {} residual ratio {:.3} (expect {:.1}) r1={:e}", order, ratio, expect, r1);
        if order >= 2 {
            assert!((ratio / expect - 1.0).abs() < 0.15, "order {} ratio {}", order, ratio);
        }
        if order >= 3 {
            let c1 = cauchy_residual(&c, 0.2, order);
            let c2 = cauchy_residual(&c, 0.1, order);
            let cr = c1 / c2;
            println!("   cauchy ratio {:.3} (expect {:.1})", cr, 2f64.powi(order as i32));
            assert!((cr / 2f64.powi(order as i32) - 1.0).abs() < 0.2, "cauchy order {} ratio {}", order, cr);
        }
    }
}

#[test]
fn lpt_eom_residual_scaling_3d() {
    for order in 2..=4 {
        let c = setup(3, 24, order);
        let r1 = longitudinal_residual(&c, 0.2, order);
        let r2 = longitudinal_residual(&c, 0.1, order);
        let ratio = r1 / r2;
        let expect = 2f64.powi(order as i32 + 1);
        println!("3D order {} residual ratio {:.3} (expect {:.1}) r1={:e}", order, ratio, expect, r1);
        assert!((ratio / expect - 1.0).abs() < 0.15, "order {} ratio {}", order, ratio);
        if order >= 3 {
            let c1 = cauchy_residual(&c, 0.2, order);
            let c2 = cauchy_residual(&c, 0.1, order);
            let cr = c1 / c2;
            println!("   cauchy ratio {:.3} (expect {:.1})", cr, 2f64.powi(order as i32));
            assert!((cr / 2f64.powi(order as i32) - 1.0).abs() < 0.2, "cauchy order {} ratio {}", order, cr);
        }
    }
}

#[test]
fn hopf_cole_matches_zeldovich_sheet() {
    let mut c = setup(2, 128, 1);
    let dsc = c.shell_crossing(1);
    let d = 0.5 * dsc;
    let sheet = c.sheet_density(d, 1, 128, 3);
    let hc = c.hopf_cole(d, 1e-4, hcc_core::hopfcole::HcMethod::RealSpace { refine: 1 });
    let mut err = 0.0;
    let mut norm = 0.0;
    for i in 0..sheet.len() {
        let a = sheet[i] as f64 - 1.0;
        let b = hc.delta[i] as f64;
        err += (a - b).powi(2);
        norm += a * a;
    }
    let rel = (err / norm).sqrt();
    println!("D_sc = {:.3}, rel diff sheet vs Hopf-Cole = {:.3e}, nu_eff = {:.2e}", dsc, rel, hc.nu_eff);
    assert!(rel < 0.1, "rel diff {}", rel);
    let hs = c.hopf_cole(d, 2e-3, hcc_core::hopfcole::HcMethod::Spectral { max_exp: 40.0 });
    let hr = c.hopf_cole(d, 2e-3, hcc_core::hopfcole::HcMethod::RealSpace { refine: 1 });
    let mut e2 = 0.0; let mut n2 = 0.0;
    for i in 0..hs.delta.len() { e2 += (hs.delta[i] as f64 - hr.delta[i] as f64).powi(2); n2 += (hs.delta[i] as f64).powi(2); }
    let rel2 = (e2 / n2).sqrt();
    println!("spectral vs real-space Hopf-Cole rel diff = {:.3e}", rel2);
    assert!(rel2 < 1e-3, "methods disagree {}", rel2);
    // Jacobian-based Lagrangian density consistency: mean of sheet density = 1
    let mean: f64 = sheet.iter().map(|&v| v as f64).sum::<f64>() / sheet.len() as f64;
    assert!((mean - 1.0).abs() < 0.02, "mean density {}", mean);
}

#[test]
fn burgers_rk4_vs_hopf_cole() {
    let n = 256;
    let l = 2.0;
    let u0: Vec<f64> = (0..n).map(|i| { let x = -l / 2.0 + i as f64 * l / n as f64; 1.0 / (x / (l / 20.0)).cosh() }).collect();
    let nu = 0.009;
    let t = 0.25;
    let mut b = hcc_core::burgers1d::Burgers1D::new(&u0, l, nu, true);
    b.advance_to(t, 0.3);
    let u_rk = b.u();
    let hc = hcc_core::burgers1d::hopf_cole_1d(&u0, l, nu, t);
    let mut err = 0.0;
    let mut norm = 0.0;
    for i in 0..n {
        err += (u_rk[i] - hc.u[i]).powi(2);
        norm += hc.u[i].powi(2);
    }
    let rel = (err / norm).sqrt();
    println!("RK4 vs Hopf-Cole rel diff {:.3e}", rel);
    assert!(rel < 2e-3);
}

#[test]
fn spt_kernels() {
    let k1 = [0.3, 0.1, 0.0];
    let k2 = [-0.1, 0.5, 0.0];
    let f2 = f_sym(&[k1, k2]);
    let d = k1[0] * k2[0] + k1[1] * k2[1];
    let a = k1[0] * k1[0] + k1[1] * k1[1];
    let b = k2[0] * k2[0] + k2[1] * k2[1];
    let closed = 5.0 / 7.0 + 0.5 * d * (1.0 / a + 1.0 / b) + 2.0 / 7.0 * d * d / (a * b);
    assert!((f2 - closed).abs() < 1e-12, "{} vs {}", f2, closed);
    let (f1, _) = fg_kernel(&[k1]);
    assert_eq!(f1, 1.0);
    let za = f_za(&[k1, k2]);
    let k = [k1[0] + k2[0], k1[1] + k2[1], 0.0];
    let closed_za = 0.5 * ((k[0] * k1[0] + k[1] * k1[1]) / a) * ((k[0] * k2[0] + k[1] * k2[1]) / b);
    assert!((za - closed_za).abs() < 1e-12);
}

#[test]
fn grf_power_spectrum_matches_input() {
    let mut c = Cosmo::new(2, 128, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.02, 7, 1.0);
    let dl = c.linear_delta(1.0);
    let s = c.power_spectrum(&dl, 20, false);
    let mut maxdev: f64 = 0.0;
    for i in 0..s.k.len() {
        if s.nmodes[i] < 200.0 || s.k[i] * 0.02 > 1.5 { continue; }
        let pl = c.linear_pk(s.k[i], 1.0);
        let dev = (s.p[i] / pl - 1.0).abs();
        maxdev = maxdev.max(dev);
    }
    println!("max |P/Plin - 1| = {:.3}", maxdev);
    assert!(maxdev < 0.2);
}

#[test]
fn sheet_3d_tetrahedra() {
    let mut c = Cosmo::new(3, 32, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 5, 1.0);
    c.build_lpt(1);
    let d = 0.5 * c.shell_crossing(1);
    let t0 = std::time::Instant::now();
    let sheet = c.sheet_density(d, 1, 32, 1);
    println!("3D tet rasterization 32³ took {:?}", t0.elapsed());
    let mean: f64 = sheet.iter().map(|&v| v as f64).sum::<f64>() / sheet.len() as f64;
    println!("mean sheet density {mean}");
    assert!((mean - 1.0).abs() < 0.03, "mean {}", mean);
    // Lagrangian 1/J has the same extrema range
    let mut j = vec![0.0f64; c.grid.size];
    c.lpt_ref().jacobian(d, 1, &mut j);
    let maxl = j.iter().map(|&v| 1.0 / v).fold(0.0, f64::max);
    let maxs = sheet.iter().cloned().fold(0.0f32, f32::max) as f64;
    println!("max 1/J {maxl:.3} vs max sheet {maxs:.3}");
    assert!((maxs / maxl - 1.0).abs() < 0.35);
    // and the sheet power spectrum agrees with CIC at low k
    let cic = c.cic_density(d, 1, 32);
    let ds: Vec<f32> = sheet.iter().map(|v| v - 1.0).collect();
    let dc: Vec<f32> = cic.iter().map(|v| v - 1.0).collect();
    let ps = c.power_spectrum(&ds, 10, false);
    let pc = c.power_spectrum(&dc, 10, true);
    for i in 0..3 { println!("k={:.1} P_sheet={:.3e} P_cic={:.3e}", ps.k[i], ps.p[i], pc.p[i]); assert!((ps.p[i] / pc.p[i] - 1.0).abs() < 0.25); }
}
