use hcc_core::direct::nufft_density;
use hcc_core::grid::Grid;
use hcc_core::fft::FftEngine;
fn main() {
    // 2D: particles on a fine grid displaced by x = q + A sin(2π q_x) (1D-like map), compare δ̂(k=2π) to exact
    let n = 64usize; let l = 1.0;
    for &(r, sigma, support) in &[(2usize, 1.25f64, 5usize), (2, 1.25, 8), (2, 1.25, 12), (2, 1.0, 8), (2, 0.8, 8), (2, 1.6, 12), (1, 1.0, 8)] {
        let nf = n * r; let dq = l / nf as f64;
        let a = 0.004 / (2.0 * std::f64::consts::PI);
        let mut pos = vec![0.0f32; nf * nf * 2];
        for i in 0..nf { for j in 0..nf { let q = i as f64 * dq; pos[(i * nf + j) * 2] = (q + a * (2.0 * std::f64::consts::PI * q).sin()) as f32; pos[(i * nf + j) * 2 + 1] = (j as f64 * dq) as f32; } }
        let rho = nufft_density(2, &pos, nf * nf, l, n, sigma, support);
        let grid = Grid::new(2, n, l); let mut eng = FftEngine::new(&grid);
        let d: Vec<f64> = rho.iter().map(|&v| v as f64 - 1.0).collect();
        let h = eng.forward_real(&d);
        // exact: 1+δ(x) = 1/(1 + 2πA cos(2πq(x))); δ̂(k=2π) ≈ -2πA/2 * N² (cos amplitude) to first order; compute exact by quadrature
        let kx = 2.0 * std::f64::consts::PI;
        let mut exact = 0.0f64; let m = 20000;
        for i in 0..m { let q = (i as f64 + 0.5) / m as f64; let x = q + a * (kx * q).sin(); // dx = (1 + 2πa cos) dq ; δ̂ = ∫ (ρ-1) e^{-ikx} dx = ∫ e^{-ikx(q)} dq - ∫ e^{-ikx} dx
            exact += (kx * x).cos() / m as f64; }
        // ∫ e^{-ikx} dx over the box = 0, so δ̂_c(k) = ∫ e^{-ikx(q)} dq (real part by symmetry); grid coefficient = δ̂_c · N²/L²
        let got = h[grid.ravel([1, 0, 0])].re / (n * n) as f64;
        println!("refine {r} sigma {sigma} support {support}: δ̂(2π)/N² = {got:.6e}, exact {exact:.6e}, ratio {:.5}", got / exact);
    }
}
