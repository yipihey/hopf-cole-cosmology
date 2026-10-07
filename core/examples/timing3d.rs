use hcc_core::ics::PkShape;
use hcc_core::Cosmo;
fn main() {
    for &(n, om) in &[(64usize, 1.0f64), (64, 0.3), (96, 0.3)] {
        let mut c = Cosmo::new(3, n, 1.0);
        c.set_cosmology(om);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.04, 3, 1.0);
        let t0 = std::time::Instant::now();
        c.build_lpt(4);
        let tb = t0.elapsed();
        let nt = c.lpt_ref().terms.len();
        let d = 0.5 * c.shell_crossing(1);
        let t1 = std::time::Instant::now();
        let s = c.sheet_density(d, 4, n, 1);
        let ts = t1.elapsed();
        let t2 = std::time::Instant::now();
        let _ = c.cic_density(d, 4, n);
        let tc = t2.elapsed();
        let mean: f64 = s.iter().map(|&v| v as f64).sum::<f64>() / s.len() as f64;
        println!("n={n} Ω_m={om}: 4LPT build {tb:?} ({nt} terms), tet sheet {ts:?} (mean {mean:.4}), CIC {tc:?}, D_sc1={d:.3}");
    }
}
