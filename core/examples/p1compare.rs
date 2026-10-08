use hcc_core::hopfcole::{dual_sheet_density, HcMethod};
use hcc_core::ics::PkShape;
use hcc_core::Cosmo;
fn rel(a: &[f32], b: &[f32]) -> f64 { let mut e = 0.0; let mut n = 0.0; for i in 0..a.len() { e += (a[i] as f64 - b[i] as f64).powi(2); n += (b[i] as f64 - 1.0).powi(2); } (e / n).sqrt() }
fn main() {
    for &n in &[256usize, 512] {
        let mut c = Cosmo::new(2, n, 1.0);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.02, 1, 1.0);
        c.build_lpt(2);
        let dsc = c.shell_crossing(2);
        for &f in &[0.5, 0.8] {
            let d = f * dsc;
            let plain = c.sheet_density(d, 2, n, 4);
            let p1 = c.sheet_density_p1(d, 2, n, 4, 1e4);
            let (hc, _) = c.hopf_cole_lpt(d, 2, 1e-5, HcMethod::RealSpace { refine: 2 }, false);
            let dual = dual_sheet_density(&c.grid, &hc.qmap);
            let fd: Vec<f32> = hc.delta.iter().map(|v| v + 1.0).collect();
            // reference: P1 sheet at 2x resolution averaged down? use high-ss P1 as reference
            let refp1 = c.sheet_density_p1(d, 2, n, 8, 1e4);
            println!("n={n} D/Dsc={f}: vs P1(ss8): plain {:.3e}, P1 {:.3e}, dual {:.3e}, FD-HC {:.3e} | plain vs dual {:.3e}, P1 vs dual {:.3e} | max plain {:.2} P1 {:.2} dual {:.2}",
                rel(&plain, &refp1), rel(&p1, &refp1), rel(&dual, &refp1), rel(&fd, &refp1), rel(&plain, &dual), rel(&p1, &dual),
                plain.iter().cloned().fold(0.0f32, f32::max), p1.iter().cloned().fold(0.0f32, f32::max), dual.iter().cloned().fold(0.0f32, f32::max));
        }
    }
}
