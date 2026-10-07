use hcc_core::ics::PkShape;
use hcc_core::lpt::{det_i_plus, mu2};
use hcc_core::spectra::{f_sym, f_za, fg_kernel};
use hcc_core::Cosmo;

fn setup(dim: usize, n: usize, order: usize) -> Cosmo {
    let mut c = Cosmo::new(dim, n, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 42, 1.0);
    c.build_lpt(order);
    c
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
                let mn = l.m[n - 1][comp][idx] as f64;
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
                let mn = l.m[n - 1][comp][idx] as f64;
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
        let m1: Vec<f64> = (0..4).map(|k| l.m[0][k][idx] as f64).collect();
        let expect = -3.0 / 7.0 * mu2(&m1, &m1, 2);
        let got = l.div[1][idx] as f64;
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
