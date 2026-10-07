fn main() {
    let n = 512; let l = 2.0;
    let u0: Vec<f64> = (0..n).map(|i| (2.0*std::f64::consts::PI*(i as f64/n as f64 - 0.5)).sin()).collect();
    let r = hcc_core::burgers1d::inviscid_1d(&u0, l, 0.2);
    for i in (0..n).step_by(32) { println!("{i:4} x={:.3} y*={:.4} rho={:.4} u={:.4} phi={:.4}", i as f64*l/n as f64, r.x0_star[i], r.rho[i], r.u[i], r.phi[i]); }
    let m: f64 = r.rho.iter().sum::<f64>() / n as f64; println!("mean rho {m}");
}
