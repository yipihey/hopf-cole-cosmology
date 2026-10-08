use hcc_core::direct::{exp_divided_difference, simplex_transform};
use hcc_core::ics::PkShape;
use hcc_core::Cosmo;
use num_complex::Complex64 as C64;

#[test]
fn divided_differences_and_simplex_transform() {
    // two nodes: (e^a - e^b)/(a - b)
    let (a, b) = (C64::new(0.0, 1.3), C64::new(0.0, -0.7));
    let dd = exp_divided_difference(&[a, b]);
    let exact = (a.exp() - b.exp()) / (a - b);
    assert!((dd - exact).norm() < 1e-14);
    // near-coincident nodes: Taylor branch must agree with the recursion just above the threshold
    let s1 = [C64::new(0.0, 2.0), C64::new(0.0, 2.06), C64::new(0.0, 2.11)];
    let s2 = [C64::new(0.0, 2.0), C64::new(0.0, 2.04), C64::new(0.0, 2.09)];
    let r1 = exp_divided_difference(&s1); // recursion (dmin 0.05 < 0.06)
    let r2 = exp_divided_difference(&s2); // Taylor
    // both must match a brute-force high-precision evaluation via the Hermite–Genocchi integral (Monte Carlo-free: use the Taylor form with many terms as truth)
    let truth = |s: &[C64]| { let m = (s[0] + s[1] + s[2]) / 3.0; let d: Vec<C64> = s.iter().map(|v| v - m).collect();
        let mut sum = C64::new(0.0, 0.0); // Σ_{a+b+c=n} d0^a d1^b d2^c / (n+2)!
        for n in 0..40usize { let mut h = C64::new(0.0, 0.0); for i in 0..=n { for j in 0..=(n - i) { let k = n - i - j; h += d[0].powu(i as u32) * d[1].powu(j as u32) * d[2].powu(k as u32); } }
            let mut f = 1.0; for q in 1..=(n + 2) { f *= q as f64; } sum += h / f; }
        m.exp() * sum };
    assert!((r1 - truth(&s1)).norm() < 1e-12, "recursion {r1} vs {}", truth(&s1));
    assert!((r2 - truth(&s2)).norm() < 1e-13, "taylor {r2} vs {}", truth(&s2));
    // triangle transform vs fine quadrature
    let v = [[0.1, 0.2, 0.0], [0.9, 0.3, 0.0], [0.4, 0.8, 0.0]];
    let k = [7.0, -3.0, 0.0];
    let ft = simplex_transform(&v, &[1.0, 1.0, 1.0], 2, k);
    let mut q = C64::new(0.0, 0.0);
    let m = 1200;
    let area2 = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
    for i in 0..m { for j in 0..m { // barycentric midpoint quadrature over the reference triangle
        let (u, w) = ((i as f64 + 0.5) / m as f64, (j as f64 + 0.5) / m as f64);
        if u + w > 1.0 { continue; }
        let x = v[0][0] + u * (v[1][0] - v[0][0]) + w * (v[2][0] - v[0][0]);
        let y = v[0][1] + u * (v[1][1] - v[0][1]) + w * (v[2][1] - v[0][1]);
        q += C64::new(0.0, -(k[0] * x + k[1] * y)).exp() * (area2.abs() / (m * m) as f64);
    } }
    println!("triangle FT {ft} vs quadrature {q}");
    assert!((ft - q).norm() < 2e-3 * ft.norm().max(1e-3));
    // P1: Σ_j λ_j = 1 ⇒ the P1 transform with all weights 1 equals the P0 one
    let ft1 = simplex_transform(&v, &[1.0, 1.0, 1.0 + 1e-12], 2, k);
    assert!((ft1 - ft).norm() < 1e-9);
    // tetrahedron: unit-ish tet at k = 0 gives its volume
    let t = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let f0 = simplex_transform(&t, &[1.0; 4], 3, [1e-9, 2e-9, -1e-9]);
    assert!((f0.re - 1.0 / 6.0).abs() < 1e-9);
}

#[test]
fn direct_spectrum_matches_exact_deposit_at_low_k() {
    let n = 64usize;
    let mut c = Cosmo::new(2, n, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 2, 1.0);
    c.build_lpt(2);
    let d = 0.6 * c.shell_crossing(2);
    let t0 = std::time::Instant::now();
    let ds = c.direct_spectrum(d, 2, false, 12, 64, 1, 1e4);
    println!("direct spectrum 64² took {:?} ({} bins)", t0.elapsed(), ds.k.len());
    let dep = c.sheet_density_exact(d, 2, n, false, 1e4);
    let dd: Vec<f32> = dep.iter().map(|v| v - 1.0).collect();
    let ps = c.power_spectrum(&dd, 12, false);
    // same lattice modes at low k (the direct method enumerates small shells completely);
    // the deposit is a cell average: divide by the top-hat window² ≈ 1 - (k dx)²/12 per axis at low k
    let dx = c.grid.dx();
    for i in 0..4 {
        let kk = ds.k[i];
        let win = (1.0 - (kk * dx).powi(2) / 24.0).powi(2); // isotropic average of Π sinc² to leading order
        let ratio = ds.p[i] / (ps.p[i] / win);
        println!("k={:.1}: direct {:.4e} deposit/W² {:.4e} ratio {:.4}", kk, ds.p[i], ps.p[i] / win, ratio);
        assert!((ratio - 1.0).abs() < 0.03, "bin {i} ratio {ratio}");
    }
}

#[test]
fn nufft_density_linear_and_refined() {
    let n = 64usize;
    let mut c = Cosmo::new(2, n, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 2, 1.0);
    c.build_lpt(2);
    // linear regime (D so small that second-order terms are < 1e-3): NUFFT P(k) ≈ D² P0
    let d = 0.004;
    let t0 = std::time::Instant::now();
    let rho = c.nufft_density(d, 1, 2, n);
    println!("NUFFT 64² refine 2 took {:?}", t0.elapsed());
    let mean: f64 = rho.iter().map(|&v| v as f64).sum::<f64>() / rho.len() as f64;
    assert!((mean - 1.0).abs() < 1e-6, "mean {mean}");
    let dl: Vec<f32> = rho.iter().map(|v| v - 1.0).collect();
    let ps = c.power_spectrum(&dl, 12, false);
    let lin = c.linear_delta(d);
    let pl = c.power_spectrum(&lin, 12, false);
    // (the Zel'dovich density differs from D δ0 by realization-specific O(D) terms, hence the loose tolerance)
    for i in 0..7 { let r = ps.p[i] / pl.p[i]; println!("k={:.1}: NUFFT/linear {:.4}", ps.k[i], r); assert!((r - 1.0).abs() < 0.03, "bin {i}: {r}"); }
    // the sharper test: at the same D the NUFFT spectrum must match the exact sheet spectrum (direct method) at low k
    let ds0 = c.direct_spectrum(d, 1, false, 12, 64, 1, 1e4);
    for i in 0..5 { let r = ps.p[i] / ds0.p[i]; println!("k={:.1}: NUFFT/direct (linear regime) {:.4}", ps.k[i], r); assert!((r - 1.0).abs() < 0.01, "bin {i}: {r}"); }
    // and at a nonlinear time the NUFFT spectrum agrees with the direct (deposit-free) sheet spectrum at low k
    let d = 0.5 * c.shell_crossing(2);
    let rn = c.nufft_density(d, 2, 2, n);
    let dn: Vec<f32> = rn.iter().map(|v| v - 1.0).collect();
    let pn = c.power_spectrum(&dn, 12, false);
    let ds = c.direct_spectrum(d, 2, false, 12, 64, 1, 1e4);
    for i in 0..4 { let r = pn.p[i] / ds.p[i]; println!("k={:.1}: NUFFT/direct {:.4}", pn.k[i], r); assert!((r - 1.0).abs() < 0.02, "bin {i}: {r}"); }
    // near shell crossing: refined NUFFT vs exact P1 sheet agree at the few-% level
    let d = 0.7 * c.shell_crossing(2);
    let t0 = std::time::Instant::now();
    let rn = c.nufft_density(d, 2, 4, n);
    println!("NUFFT 64² refine 4 took {:?}", t0.elapsed());
    let ex = c.sheet_density_exact(d, 2, n, true, 1e4);
    let mut e = 0.0; let mut nn = 0.0;
    for i in 0..rn.len() { e += (rn[i] as f64 - ex[i] as f64).powi(2); nn += (ex[i] as f64 - 1.0).powi(2); }
    println!("NUFFT vs exact P1 rel rms {:.3e}", (e / nn).sqrt());
    assert!((e / nn).sqrt() < 0.08);
}
