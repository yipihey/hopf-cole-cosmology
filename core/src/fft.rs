//! Thin multi-dimensional FFT layer on top of rustfft (complex f64).

use crate::grid::Grid;
use num_complex::Complex64 as C64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct FftEngine {
    pub grid: Grid,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    scratch: Vec<C64>,
}

impl FftEngine {
    pub fn new(grid: &Grid) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(grid.n);
        let inv = planner.plan_fft_inverse(grid.n);
        let scratch = vec![C64::new(0.0, 0.0); fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
        FftEngine { grid: grid.clone(), fwd, inv, scratch }
    }

    fn along_axis(&mut self, data: &mut [C64], axis: usize, forward: bool) {
        let n = self.grid.n;
        let plan = if forward { self.fwd.clone() } else { self.inv.clone() };
        let dim = self.grid.dim;
        // stride of this axis and number of lines
        let stride = n.pow((dim - 1 - axis) as u32);
        if stride == 1 {
            for chunk in data.chunks_mut(n) {
                plan.process_with_scratch(chunk, &mut self.scratch);
            }
        } else {
            let mut line = vec![C64::new(0.0, 0.0); n];
            let block = stride * n; // size of a block spanned by this axis
            let nblocks = data.len() / block;
            for b in 0..nblocks {
                for s in 0..stride {
                    let base = b * block + s;
                    for i in 0..n {
                        line[i] = data[base + i * stride];
                    }
                    plan.process_with_scratch(&mut line, &mut self.scratch);
                    for i in 0..n {
                        data[base + i * stride] = line[i];
                    }
                }
            }
        }
    }

    /// In-place forward FFT (unnormalized).
    pub fn forward(&mut self, data: &mut [C64]) {
        for axis in 0..self.grid.dim {
            self.along_axis(data, axis, true);
        }
    }
    /// In-place inverse FFT, normalized by 1/size.
    pub fn inverse(&mut self, data: &mut [C64]) {
        for axis in 0..self.grid.dim {
            self.along_axis(data, axis, false);
        }
        let norm = 1.0 / self.grid.size as f64;
        for v in data.iter_mut() {
            *v *= norm;
        }
    }
    pub fn forward_real(&mut self, f: &[f64]) -> Vec<C64> {
        let mut c: Vec<C64> = f.iter().map(|&x| C64::new(x, 0.0)).collect();
        self.forward(&mut c);
        c
    }
    pub fn forward_real_f32(&mut self, f: &[f32]) -> Vec<C64> {
        let mut c: Vec<C64> = f.iter().map(|&x| C64::new(x as f64, 0.0)).collect();
        self.forward(&mut c);
        c
    }
    pub fn inverse_to_real(&mut self, mut c: Vec<C64>) -> Vec<f64> {
        self.inverse(&mut c);
        c.iter().map(|v| v.re).collect()
    }
    pub fn inverse_to_real_f32(&mut self, mut c: Vec<C64>) -> Vec<f32> {
        self.inverse(&mut c);
        c.iter().map(|v| v.re as f32).collect()
    }
}

/// Spectral gradient component a: returns IFFT(i k_a f_hat) as real field.
pub fn gradient_component(eng: &mut FftEngine, fhat: &[C64], a: usize) -> Vec<f64> {
    let g = eng.grid.clone();
    let mut d: Vec<C64> = fhat.iter().enumerate().map(|(idx, &v)| {
        let k = g.kvec(idx);
        C64::new(0.0, k[a]) * v
    }).collect();
    eng.inverse(&mut d);
    d.iter().map(|v| v.re).collect()
}

/// Spectral second derivative ∂_a ∂_b f: IFFT(-k_a k_b f_hat).
pub fn hessian_component(eng: &mut FftEngine, fhat: &[C64], a: usize, b: usize) -> Vec<f64> {
    let g = eng.grid.clone();
    let mut d: Vec<C64> = fhat.iter().enumerate().map(|(idx, &v)| {
        let k = g.kvec(idx);
        v * (-k[a] * k[b])
    }).collect();
    eng.inverse(&mut d);
    d.iter().map(|v| v.re).collect()
}

/// Inverse Laplacian in Fourier space (zero mode set to zero): returns -f_hat/k^2.
pub fn inv_laplacian_hat(grid: &Grid, fhat: &[C64]) -> Vec<C64> {
    fhat.iter().enumerate().map(|(idx, &v)| {
        let k2 = grid.k2(idx);
        if k2 == 0.0 { C64::new(0.0, 0.0) } else { -v / k2 }
    }).collect()
}
