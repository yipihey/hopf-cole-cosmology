// catalog.js - the selectable content of the 2D field panels: labels, colour
// handling, captions and data providers. Pure data + small functions.

import { fmap, percentiles, mean, minmax } from '../hcc.js';

export const LIN_MAX = 5;       // upper colour limit for non-log density maps

const SUB_OF = [['sheet', 'sheet'], ['cic', 'CIC'], ['hc', 'Hopf–Cole'], ['hcdual', 'Hopf–Cole dual sheet'], ['lin', 'linear']];
const OF_NAME = { sheet: 'sheet', cic: 'CIC', hc: 'Hopf–Cole', hcdual: 'Hopf–Cole dual sheet', lin: 'linear' };

/** Shift the raw FFT-ordered n×n map [ikx*n+iky] so that k = 0 sits at the centre. */
export function fftshift2D(a, n) {
  const out = new Float32Array(n * n), h = n >> 1;
  for (let i = 0; i < n; i++) {
    const ii = (i + h) % n;
    for (let j = 0; j < n; j++) out[ii * n + (j + h) % n] = a[i * n + j];
  }
  return out;
}

/**
 * kind -> definition.
 *  cls   : 'density' | 'sym' | 'auto' | 'amp' | 'phase' | 'psihat'  (colour range policy)
 *  cmap  : default colormap; log: default log flag (density only)
 *  subs  : sub-selector spec {label, options:[[value,label]], def}
 *  data(eng, P, sub) -> Float32Array in the [ix*n+iy] layout of the WASM
 *  caption(sub) -> string
 */
export const CATALOG = {
  sheetgpu: {
    label: 'Sheet density (GPU, additive triangles)', cls: 'density', cmap: 'magma', log: true, unit: 'ρ/ρ̄', gpu: true,
    subs: { label: 'draw', options: [['density', 'density'], ['wire', 'wire'], ['both', 'both']], def: 'both' },
    caption: () => 'The Lagrangian grid x(q,D) = q + ΣDⁿΨ⁽ⁿ⁾(q) drawn as triangles whose mass is spread over their image and added: folded, multi-stream regions accumulate. “wire” shows the deformed grid; folds appear after shell crossing.',
  },
  sheetcpu: {
    label: 'Sheet density (CPU rasterized)', cls: 'density', cmap: 'magma', log: true, unit: 'ρ/ρ̄',
    data: (e, P) => e.sheet(P),
    caption: () => 'Exact sheet density Σ 1/|J| over all streams, point-sampled with 2×2 supersampling in the Rust core. Same quantity as the GPU panel.',
  },
  cic: {
    label: 'CIC density', cls: 'density', cmap: 'magma', log: true, unit: 'ρ/ρ̄',
    data: (e, P) => e.cic(P),
    caption: () => 'Cloud-in-cell deposit of the N² displaced particles: the usual N-body estimate. Noisier than the sheet and blurred by the CIC window.',
  },
  hc: {
    label: 'Hopf–Cole density', cls: 'density', cmap: 'magma', log: true, unit: '1+δ',
    data: (e, P) => fmap(e.hc(P).delta, (v) => v + 1),
    caption: (sub, P) => (P && P.hs && P.hs !== 'zel' && P.order > 1
      ? `1+δ = det(I − ∇∇Φ) with Φ the Legendre transform (Hopf–Lax minimum, heat-kernel smoothed at finite ν) of the order-${P.order} ${P.hs === 'lptT' && P.order > 2 ? 'longitudinal + transverse-corrected ' : 'longitudinal '}Lagrangian map. Before shell crossing it is the nLPT density, particle-free on the Eulerian grid; afterwards it keeps one stream per point (adhesion) with shocks of width ~√(νD).`
      : '1+δ = det(I − D∇∇Φ_v) from the viscous-Burgers (adhesion) solution obtained with the Hopf–Cole transform. Before shell crossing it equals the Zel’dovich sheet; afterwards mass sticks into shocks of width ~√(νD).'),
  },
  hcdual: {
    label: 'Hopf–Cole dual sheet (mass-conserving)', cls: 'density', cmap: 'magma', log: true, unit: 'ρ/ρ̄',
    data: (e, P) => e.dualRho(P),
    caption: () => 'The mass of an Eulerian cell is the Lagrangian volume of its preimage under the Hopf–Cole inverse map q(x) = x − D∇Φ_v (the polygon spanned by q at the cell’s corners), so mass is conserved exactly and peaks are not under-resolved by finite-difference Hessians; before shell crossing it should beat both the finite-difference Hopf–Cole density (in peaks) and the forward sheet (in voids).',
  },
  lin: {
    label: 'Linear density D δ0', cls: 'sym', cmap: 'rdbu', unit: 'δ_lin',
    data: (e, P) => e.linear(P),
    caption: () => 'Linear theory δ_lin = D δ0 (signed, mean 0). Gaussian, with no caustics or voids; the reference for the cross-correlation r(k).',
  },
  phi: {
    label: 'Velocity potential Φ_v (Hopf–Cole)', cls: 'sym', cmap: 'rdbu', unit: 'Φ_v',
    data: (e, P) => { const f = e.hc(P).phi; const m = mean(f); return fmap(f, (v) => v - m); },
    caption: () => 'Φ_v(x,D) = −2ν ln ψ with velocity u = ∇Φ_v (mean removed). As ν→0 it is the Hopf–Lax minimum over Lagrangian positions, with kinks where shocks form.',
  },
  lnpsi: {
    label: 'ln ψ', cls: 'auto', cmap: 'viridis', unit: 'ln ψ − max',
    data: (e, P) => { const f = e.hc(P).lnpsi; const m = minmax(f)[1]; return fmap(f, (v) => v - m); },
    caption: () => 'ln ψ with ψ = exp(−Φ_v/2ν), which obeys the linear heat equation ∂_Dψ = ν∇²ψ. Its dynamic range (Φmax − Φmin)/2ν is why the naive Fourier solution fails at small ν.',
  },
  invj: {
    label: 'Lagrangian 1/J', cls: 'density', cmap: 'magma', log: true, unit: '1/J', lagr: true,
    data: (e, P) => fmap(e.jacobian(P), (j) => (j > 1e-6 ? 1 / j : 1e6)),
    caption: () => 'Density carried by each fluid element, 1/J(q) with J = det ∂x/∂q at the chosen LPT order, plotted at its initial position q. Elements with J ≤ 0 have shell-crossed (shown at maximum).',
  },
  lptsrc: {
    label: 'LPT source ∇·Ψ⁽ⁿ⁾', cls: 'sym', cmap: 'rdbu', unit: '∇·Ψ', lagr: true,
    subs: { label: 'n', options: [['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], def: '2' },
    data: (e, P, sub) => e.div(P, Number(sub)),
    caption: (sub) => `Longitudinal source ∇·Ψ⁽${sub}⁾ of the order-${sub} displacement on the Lagrangian grid` + (sub === '1' ? ' (equal to −δ0).' : ', built recursively from products of lower-order displacement gradients (μ₂, μ₃ invariants).'),
  },
  lptcurl: {
    label: 'LPT curl source (n≥3)', cls: 'sym', cmap: 'rdbu', unit: '(∇×Ψ)_z', lagr: true,
    subs: { label: 'n', options: [['3', '3'], ['4', '4']], def: '3' },
    data: (e, P, sub) => e.curl(P, Number(sub)),
    caption: (sub) => `Transverse part (∇×Ψ⁽${sub}⁾)_z, the z-component in 2D. Displacements of first and second order are curl-free; vorticity of the Lagrangian map first appears at third order.`,
  },
  fabs: {
    label: '|δ̂(k)| of …', cls: 'amp', cmap: 'viridis', unit: 'log₁₀|δ̂|/max', fourier: true,
    subs: { label: 'of', options: SUB_OF, def: 'sheet' },
    data: (e, P, sub) => e.fmaps(sub, P).amp,
    caption: (sub) => `log₁₀|δ̂(k)|/max of the ${OF_NAME[sub] || sub} density; k_x horizontal, k_y vertical, k = 0 at the centre. Nonlinear evolution fills in new modes: harmonics and sums/differences of the linear ones.`,
  },
  fphase: {
    label: 'phase of δ̂(k) of …', cls: 'phase', cmap: 'twilight', unit: 'arg δ̂ [rad]', fourier: true,
    subs: { label: 'of', options: SUB_OF, def: 'sheet' },
    data: (e, P, sub) => e.fmaps(sub, P).phase,
    caption: (sub) => `Phase arg δ̂(k) of the ${OF_NAME[sub] || sub} density. Linear growth leaves the initial phases untouched; mode coupling makes the phases of generated modes depend on those of their parents.`,
  },
  psihat: {
    label: '|ψ̂(k)|', cls: 'psihat', cmap: 'viridis', unit: 'log₁₀|ψ̂|/max', fourier: true,
    data: (e, P) => e.hc(P).psihat, // already fft-shifted [ikx*n+iky]
    caption: () => 'log₁₀|ψ̂(k)|/max of ψ = exp(−Φ_v/2ν) at time D (heat-equation solution). The multiplier exp(−νk²D) acts here; the floor of this map shows the numerical dynamic range.',
  },
  speed: {
    label: 'Speed |u| (Hopf–Cole)', cls: 'auto', cmap: 'viridis', unit: '|u|',
    data: (e, P) => { const v = e.hc(P).vel, n2 = v.length / 2, o = new Float32Array(n2); for (let i = 0; i < n2; i++) o[i] = Math.hypot(v[2 * i], v[2 * i + 1]); return o; },
    caption: () => 'Speed |u| = |∇Φ_v| of the Burgers flow. In the adhesion picture matter streams along u until it hits a shock.',
  },
};

export const KIND_ORDER = ['sheetgpu', 'sheetcpu', 'cic', 'hc', 'hcdual', 'lin', 'phi', 'lnpsi', 'invj', 'lptsrc', 'lptcurl', 'fabs', 'fphase', 'psihat', 'speed'];

/** Percentile-based range helpers for special classes. */
export function psihatRange(arr) {
  const [lo] = percentiles(arr, 0.5, 99.5);
  return [Math.min(lo, -1), 0];
}
