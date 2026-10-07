use hcc_core::hopfcole::HcMethod;
use hcc_core::ics::Preset;
use hcc_core::Cosmo;
// δ0 = A cos(k x): Zel'dovich x = q + D Ψ1, Ψ1 = -dϕ/dx, ϕ = -A cos(kq)/k² → Ψ1 = -A sin(kq)/k
// 1+δ(x) = 1/(1 - D A cos(k q(x)))  (exact before shell crossing at D A = 1)
fn main() {
    for &n in &[128usize, 256] {
        let mut c = Cosmo::new(2, n, 1.0);
        let m = 2.0;
        c.set_ic_preset(Preset::PlaneWaves { modes: vec![([2, 0, 0], 1.0, 0.0)] }, 0.0, 0.0);
        let a = 1.0; // amplitude of cos (rms = 1/√2)
        let k = 2.0 * std::f64::consts::PI * m;
        let d = 0.6;
        c.build_lpt(1);
        let sheet = c.sheet_density(d, 1, n, 4);
        for (nu, r) in [(1e-4, 1), (1e-5, 1), (1e-5, 2), (1e-6, 2), (1e-6, 4), (3e-7, 4)] {
            let t0 = std::time::Instant::now();
            let hc = c.hopf_cole(d, nu, HcMethod::RealSpace { refine: r });
            print!("refine={r} {:?} ", t0.elapsed());
            let mut err = 0.0; let mut errs = 0.0; let mut norm = 0.0; let mut maxe: f64 = 0.0;
            for i in 0..n { // along x (row index i), any column
                let x = i as f64 / n as f64;
                // solve x = q - D A sin(kq)/k for q by Newton
                let mut q = x;
                for _ in 0..50 { let f = q - d * a * (k * q).sin() / k - x; let fp = 1.0 - d * a * (k * q).cos(); q -= f / fp; }
                let exact = 1.0 / (1.0 - d * a * (k * q).cos()) - 1.0;
                let got = hc.delta[i * n] as f64;
                let gs = sheet[i * n] as f64 - 1.0;
                err += (got - exact).powi(2); errs += (gs - exact).powi(2); norm += exact * exact; maxe = maxe.max((got - exact).abs());
            }
            println!("n={n} nu={nu:.0e}: HC rel err {:.2e} (max abs {:.2e}), sheet rel err {:.2e}", (err / norm).sqrt(), maxe, (errs / norm).sqrt());
        }
    }
}
