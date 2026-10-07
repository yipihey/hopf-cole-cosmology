fn main() {
    let n = 512; let l = 1.0;
    let x: Vec<f64> = (0..n).map(|i| { let q = i as f64 / n as f64; q + 0.1 * (2.0*std::f64::consts::PI*q).sin() }).collect();
    let rho = hcc_core::burgers1d::sheet_density_1d(&x, l, n);
    let mut maxerr: f64 = 0.0;
    for i in 0..n { let xe = (i as f64 + 0.5) / n as f64; // find q with x(q)=xe by Newton
        let mut q = xe; for _ in 0..30 { let f = q + 0.1*(2.0*std::f64::consts::PI*q).sin() - xe; let fp = 1.0 + 0.2*std::f64::consts::PI*(2.0*std::f64::consts::PI*q).cos(); q -= f/fp; }
        let exact = 1.0/(1.0 + 0.2*std::f64::consts::PI*(2.0*std::f64::consts::PI*q).cos());
        maxerr = maxerr.max((rho[i]-exact).abs()/exact);
        if i % 64 == 0 { println!("{i} rho={:.4} exact={:.4}", rho[i], exact); }
    }
    println!("max rel err {maxerr:.2e}, mean {:.6}", rho.iter().sum::<f64>()/n as f64);
}
