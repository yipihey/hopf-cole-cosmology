//! Periodic Cartesian grid in 1, 2 or 3 dimensions with Fourier bookkeeping.
//!
//! Row-major storage: the *last* index is fastest.  For a field `f` of shape
//! `[n, n, n]`, element `(i, j, k)` is at `f[(i*n + j)*n + k]`.

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Grid {
    pub dim: usize,
    pub n: usize,
    pub l: f64,
    pub size: usize,
}

impl Grid {
    pub fn new(dim: usize, n: usize, l: f64) -> Self {
        assert!(dim >= 1 && dim <= 3);
        Grid { dim, n, l, size: n.pow(dim as u32) }
    }
    #[inline]
    pub fn dx(&self) -> f64 {
        self.l / self.n as f64
    }
    /// Fundamental wavenumber 2π/L.
    #[inline]
    pub fn kf(&self) -> f64 {
        2.0 * PI / self.l
    }
    /// Nyquist wavenumber π/dx.
    #[inline]
    pub fn knyq(&self) -> f64 {
        PI / self.dx()
    }
    /// Integer frequency index (fftfreq convention) for grid index `i`.
    #[inline]
    pub fn ifreq(&self, i: usize) -> i64 {
        if i < self.n / 2 { i as i64 } else { i as i64 - self.n as i64 }
    }
    /// Wavenumber component for grid index `i`.  The Nyquist index is kept
    /// (its magnitude is used in |k|), but see `kd` for derivatives.
    #[inline]
    pub fn k1(&self, i: usize) -> f64 {
        self.kf() * self.ifreq(i) as f64
    }
    /// Wavenumber component to use inside *odd* derivative operators: the
    /// Nyquist mode is zeroed so that i*k*f stays Hermitian (real output).
    #[inline]
    pub fn kd(&self, i: usize) -> f64 {
        if self.n % 2 == 0 && i == self.n / 2 { 0.0 } else { self.k1(i) }
    }
    /// Decompose a flat index into per-axis indices.
    #[inline]
    pub fn unravel(&self, idx: usize) -> [usize; 3] {
        match self.dim {
            1 => [idx, 0, 0],
            2 => [idx / self.n, idx % self.n, 0],
            _ => [idx / (self.n * self.n), (idx / self.n) % self.n, idx % self.n],
        }
    }
    #[inline]
    pub fn ravel(&self, ijk: [usize; 3]) -> usize {
        match self.dim {
            1 => ijk[0],
            2 => ijk[0] * self.n + ijk[1],
            _ => (ijk[0] * self.n + ijk[1]) * self.n + ijk[2],
        }
    }
    /// Wavevector (derivative-safe components) at flat index.
    #[inline]
    pub fn kvec(&self, idx: usize) -> [f64; 3] {
        let ijk = self.unravel(idx);
        let mut k = [0.0; 3];
        for a in 0..self.dim {
            k[a] = self.kd(ijk[a]);
        }
        k
    }
    /// |k|^2 using the full (Nyquist-including) components.
    #[inline]
    pub fn k2(&self, idx: usize) -> f64 {
        let ijk = self.unravel(idx);
        let mut s = 0.0;
        for a in 0..self.dim {
            let k = self.k1(ijk[a]);
            s += k * k;
        }
        s
    }
    /// Shape array (unused axes = 1).
    pub fn shape(&self) -> [usize; 3] {
        let mut s = [1usize; 3];
        for a in 0..self.dim {
            s[a] = self.n;
        }
        s
    }
}
