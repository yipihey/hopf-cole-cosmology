use hcc_core::ics::PkShape;
use hcc_core::Cosmo;
fn stats(name: &str, v: &[f32]) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{:>12}: mean {:.4e} rms {:.4e} min {:.4e} max {:.4e}", name, mean, var.sqrt(), mn, mx);
}
fn main() {
    let mut c = Cosmo::new(2, 128, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.06, 42, 1.0);
    c.build_lpt(1);
    let dsc = c.shell_crossing(1);
    let d = 0.5 * dsc;
    println!("D_sc={dsc:.4} D={d:.4}");
    let dl = c.linear_delta(d);
    stats("lin delta", &dl);
    let phi: Vec<f32> = c.phi0.iter().map(|&v| v as f32).collect();
    stats("phi0", &phi);
    let pos = c.positions(d, 1);
    let disp: Vec<f32> = (0..c.grid.size).map(|i| { let ijk = c.grid.unravel(i); pos[2*i] - ijk[0] as f32 * c.grid.dx() as f32 }).collect();
    stats("disp x", &disp);
    let sheet = c.sheet_density(d, 1, 128, 3);
    stats("sheet rho", &sheet);
    let cic = c.cic_density(d, 1, 128);
    stats("cic rho", &cic);
    let mut j = vec![0.0f64; c.grid.size];
    c.lpt_ref().jacobian(d, 1, &mut j);
    let jl: Vec<f32> = j.iter().map(|&v| (1.0 / v) as f32).collect();
    stats("1/J (Lagr)", &jl);
    for nu in [1e-3, 1e-4, 1e-5, 1e-6] {
        let t0 = std::time::Instant::now();
        let hc = c.hopf_cole(d, nu, hcc_core::hopfcole::HcMethod::RealSpace { refine: 1 });
        println!("  realspace took {:?}", t0.elapsed());
        println!("nu={nu:e} nu_eff={:e} exp range={:.1}", hc.nu_eff, hc.exponent_range);
        stats("hc delta", &hc.delta);
        stats("hc phi_v", &hc.phi_v);
        stats("hc vel x", &hc.velocity.iter().step_by(2).cloned().collect::<Vec<f32>>());
        stats("hc lnpsi", &hc.lnpsi);
        let mut err = 0.0; let mut norm = 0.0;
        for i in 0..sheet.len() { let a = sheet[i] as f64 - 1.0; let b = hc.delta[i] as f64; err += (a-b).powi(2); norm += a*a; }
        println!("  rel diff vs sheet: {:.3e}", (err/norm).sqrt());
    }
    let vel = c.velocities(d, 1);
    stats("lpt vel x", &vel.iter().step_by(2).cloned().collect::<Vec<f32>>());
}
