// Shared helpers around the hcc-core WASM module.
// Loaded from both the chapter widgets and the standalone lab.

let _mod = null;
let _loading = null;

/** Load (once) and return the wasm module namespace. */
export async function loadCore() {
  if (_mod) return _mod;
  if (!_loading) {
    _loading = (async () => {
      const url = new URL('../pkg/hcc_core.js', import.meta.url);
      const mod = await import(url.href);
      await mod.default({ module_or_path: new URL('../pkg/hcc_core_bg.wasm', import.meta.url) });
      _mod = mod;
      return mod;
    })();
  }
  return _loading;
}

/** Unpack the flat [k..., P..., N...] arrays returned by power_spectrum / cross_spectrum. */
export function unpackSpectrum(flat) {
  const nb = flat.length / 3;
  return { k: flat.slice(0, nb), p: flat.slice(nb, 2 * nb), n: flat.slice(2 * nb, 3 * nb) };
}

/** Unpack the flat [Plin..., P22..., P13...] arrays returned by one_loop. */
export function unpackLoop(flat) {
  const nb = flat.length / 3;
  return { plin: flat.slice(0, nb), p22: flat.slice(nb, 2 * nb), p13: flat.slice(2 * nb, 3 * nb) };
}

/** Linear grid of x positions for an N-point periodic box of length L starting at x0. */
export function linspaceBox(n, L, x0 = 0) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = x0 + (i * L) / n;
  return x;
}

/** Positive wavenumbers k_j = 2π j / L for j = 0..n/2. */
export function kaxis(n, L) {
  const m = Math.floor(n / 2) + 1;
  const k = new Float64Array(m);
  for (let j = 0; j < m; j++) k[j] = (2 * Math.PI * j) / L;
  return k;
}

/** First half (non-negative frequencies) of a full-length spectrum array. */
export function halfSpectrum(full) {
  const m = Math.floor(full.length / 2) + 1;
  return full.slice(0, m);
}

/** Statistics helpers. */
export function minmax(a) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}
export function percentiles(a, plo = 0.5, phi = 99.5) {
  const n = a.length;
  const step = Math.max(1, Math.floor(n / 50000));
  const s = [];
  for (let i = 0; i < n; i += step) if (Number.isFinite(a[i])) s.push(a[i]);
  s.sort((x, y) => x - y);
  const q = (p) => s[Math.min(s.length - 1, Math.max(0, Math.floor((p / 100) * (s.length - 1))))];
  return [q(plo), q(phi)];
}
export function rms(a) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / a.length);
}
export function mean(a) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
/** Elementwise map producing a Float32Array. */
export function fmap(a, f) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = f(a[i]);
  return out;
}
/** Breaking time t* = -1/min(u0') for a periodic profile sampled with spacing dx. */
export function breakingTime(u0, dx) {
  const n = u0.length;
  let mn = Infinity;
  for (let i = 0; i < n; i++) {
    const d = (u0[(i + 1) % n] - u0[(i + n - 1) % n]) / (2 * dx);
    if (d < mn) mn = d;
  }
  return mn < 0 ? -1 / mn : Infinity;
}

/**
 * Least-squares fit of the EFT counterterm: P_meas - P_1loop = -2 c_s^2 k^2 P_lin for k < kmax.
 * Returns {cs2, chi2}.
 */
export function fitCounterterm(k, pMeas, pLoop, pLin, kmax) {
  let num = 0, den = 0;
  for (let i = 0; i < k.length; i++) {
    if (!(k[i] < kmax) || !Number.isFinite(pMeas[i]) || !Number.isFinite(pLoop[i])) continue;
    const y = pMeas[i] - pLoop[i];
    const x = -2 * k[i] * k[i] * pLin[i];
    num += x * y; den += x * x;
  }
  const cs2 = den > 0 ? num / den : 0;
  let chi2 = 0;
  for (let i = 0; i < k.length; i++) {
    if (!(k[i] < kmax)) continue;
    const r = (pMeas[i] - pLoop[i] + 2 * cs2 * k[i] * k[i] * pLin[i]) / Math.max(1e-30, pLin[i]);
    chi2 += r * r;
  }
  return { cs2, chi2 };
}

/** Standard initial profiles for the 1D widgets on x ∈ [-L/2, L/2). */
export const PROFILES = {
  sech:  { label: 'sech pulse (notebook)', f: (x, L) => 1 / Math.cosh(x / (L / 20)) },
  sech2: { label: 'sech² pulse (exercise)', f: (x, L) => 1 / Math.cosh(x / (L / 10)) ** 2 },
  sine:  { label: 'sine wave', f: (x, L) => Math.sin(2 * Math.PI * x / L) },
  step:  { label: 'smoothed step', f: (x, L) => 0.5 * (1 + Math.tanh(-x / (L / 60))) * Math.cos(Math.PI * x / L) ** 2 },
  twoSine: { label: 'two waves', f: (x, L) => Math.sin(2 * Math.PI * x / L) + 0.5 * Math.sin(6 * Math.PI * x / L + 1) },
};
export function sampleProfile(key, n, L) {
  const u = new Float64Array(n);
  const f = PROFILES[key].f;
  for (let i = 0; i < n; i++) u[i] = f(-L / 2 + (i * L) / n, L);
  return u;
}

/** Deterministic pseudo-random Gaussian 1D field with power ∝ k^n, smoothed by exp(-k²R²/2). */
export function randomField1D(n, L, seed, slope = -1, R = 0) {
  // xorshift32
  let s = (seed | 0) || 1;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) / 4294967296); };
  const out = new Float64Array(n);
  const nmodes = Math.floor(n / 2);
  for (let j = 1; j <= nmodes; j++) {
    const k = 2 * Math.PI * j / L;
    const amp = Math.sqrt(Math.pow(k, slope)) * Math.exp(-0.5 * k * k * R * R);
    const u1 = rnd(), u2 = rnd();
    const g = Math.sqrt(-2 * Math.log(Math.max(1e-12, u1)));
    const a = amp * g * Math.cos(2 * Math.PI * u2), b = amp * g * Math.sin(2 * Math.PI * u2);
    for (let i = 0; i < n; i++) {
      const x = (i * L) / n;
      out[i] += a * Math.cos(k * x) + b * Math.sin(k * x);
    }
  }
  const r = rms(out) || 1;
  for (let i = 0; i < n; i++) out[i] /= r;
  return out;
}

/** Transpose an n×n field stored as [ix*n + iy] into [iy*n + ix] (so x runs horizontally in FieldView). */
export function transpose2D(a, n) {
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[j * n + i] = a[i * n + j];
  return out;
}
/** Extract a 2D slice (as [row*n+col]) from a 3D field [(i*n+j)*n+k]; axis 0,1,2 = x,y,z. */
export function slice3D(a, n, axis, index) {
  const out = new Float32Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    let i, j, k;
    if (axis === 2) { i = c; j = r; k = index; }       // show x horizontal, y vertical
    else if (axis === 1) { i = c; j = index; k = r; }  // x horizontal, z vertical
    else { i = index; j = c; k = r; }                  // y horizontal, z vertical
    out[r * n + c] = a[(i * n + j) * n + k];
  }
  return out;
}
