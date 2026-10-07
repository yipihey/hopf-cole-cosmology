use hcc_core::ics::PkShape;
use hcc_core::spectra::{one_loop, Kernels};
use hcc_core::Cosmo;
fn main() {
    let mut c = Cosmo::new(2, 256, 1.0);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.02, 1, 1.0);
    let pk = c.pk.clone().unwrap();
    let f = |k: f64| pk.eval(k);
    let kf = c.grid.kf();
    let d = 0.3f64;
    for &qmin in &[0.05 * kf, 0.9 * kf] {
        println!("qmin = {:.3} kf", qmin / kf);
        for &k in &[10.0, 20.0, 40.0, 80.0, 160.0] {
            let (p22, p13) = one_loop(&f, 2, k, Kernels::Spt, qmin, c.grid.knyq() * 4.0, 120, 64);
            let pl = d * d * f(k);
            println!("  k={k:6.1} Plin={pl:.3e} P22={:.3e} P13={:.3e} 1loop/lin={:.3}", d.powi(4) * p22, d.powi(4) * p13, (pl + d.powi(4) * (p22 + p13)) / pl);
        }
    }
}
