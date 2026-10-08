// direct.js - deposit-free ("direct") spectrum of the phase-space sheet on the GPU (port of core/src/direct.rs::direct_spectrum).
//
// The Fourier transform of the piecewise-constant (P0) or piecewise-linear (P1) sheet density at the lattice mode k = 2 pi m is a
// sum over the simplices T (two triangles per Lagrangian cell in 2D, six Kuhn tetrahedra in 3D, periodic +L wrap) of
//
//    P0:  dx^d  exp[s_0, .., s_d]                       s_j = -i k.v_j       (divided difference of the exponential)
//    P1:  dx^d  sum_j w'_j exp[s_0, .., s_d, s_j]       w'_j = w_j / mean(w)  (one repeated node)
//
// (Hermite-Genocchi: int_T e^{-ik.x} d^dx = d! |T| exp[s_0..s_d]; the simplex mass m_T/|T| cancels the volume.)  P(k) = |delta^(k)|^2 / L^d
// (L = 1), averaged per |k| bin over `perBin` sampled lattice modes (all modes of a shell when it is small), exactly like direct_spectrum.
//
//   const ds = new GpuDirectSpectrum(device); await ds.init();
//   const r  = await ds.spectrum({ dim, n, pos, w, p1, nbins, perBin, seed });   // {k, p, n, nmodes, ms}
//   const dh = await ds.transform({ dim, n, pos, w, p1, modes });              // per-mode complex delta^: Float32Array(2 nm) + ms
//
// `pos`: GPUBuffer, interleaved positions x = q + Psi of the n^dim Lagrangian vertices in box units (unwrapped; index (ix*n+iy)*n+iz),
// e.g. GpuCosmo3D.positions().  `w`: GPUBuffer of the n^dim vertex densities 1/|J| (P1) or null.  n must be a power of two.
//
// P1 in 3D shares one divided-difference table of the four distinct vertex nodes between the four repeated-node sums (only the entries that
// contain both copies of the repeated node are new); in 2D the plain per-vertex evaluation is faster.
//
// Layout: one thread per (mode, chunk of Lagrangian cells).  A workgroup holds 64 consecutive modes of the same chunk, so all threads read
// the same vertices (broadcast) and keep their complex partial sum in registers.  Stage 1 writes the partial sums [mode][chunk] (each thread
// owns its slot, so submits can accumulate into it without atomics); stage 2 (one workgroup per mode) tree-reduces the chunks.  Large problems
// are split into several submits, each covering a slice of every chunk (GPU watchdog).
//
// f32 strategy.  The phases s_j = -k.v_j reach 10^3 rad, so they are never formed: the lattice part of m.x is exact integer arithmetic
// ((m.i) mod n)/n, the displacement part m.Psi is taken in turns, and all node DIFFERENCES (the denominators of the divided differences and
// the arguments of the exponentials) come from the edge vectors, t_j = m.(v_j - v_0), reduced with fract() before sin/cos.  Divided
// differences over the sorted nodes: a Newton table where an entry spans at least THR = 0.5 rad, and the Taylor series about the mean of
// its own nodes (complete homogeneous polynomials h_m of i psi = i^m h_m(psi): real recursion) where it spans less - so clusters, exactly
// coincident nodes (edges perpendicular to k, the repeated P1 node) and tiny simplices stay accurate however wide the rest of the simplex
// is stretched in phase.  (core/src/direct.rs applies one Taylor series to the whole node set whenever any pair is closer than 0.05, which
// is converged to f64 roundoff for the spreads met here, but would lose f32 digits to cancellation.)

import { ParamRing, makePipeline, makeBindGroup, compile, readF32, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';

export const DIRECT_THR = 0.5;       // phase span (rad) below which a divided-difference entry is evaluated by its local Taylor series

// ---------------------------------------------------------------------------------------------------------------------------------
// mode sampling: a faithful port of direct.rs::sample_modes with the same SplitMix64 / xoshiro256** stream (so the lattice vectors of
// the sampled bins equal those of the WASM reference for the same seed)

const M64 = (1n << 64n) - 1n;
const rotl = (x, k) => ((x << k) | (x >> (64n - k))) & M64;
export class Rng {
  constructor(seed) {
    let z = BigInt(seed) & M64;
    this.s = [0n, 0n, 0n, 0n];
    for (let i = 0; i < 4; i++) {
      z = (z + 0x9E3779B97F4A7C15n) & M64;
      let x = z;
      x = ((x ^ (x >> 30n)) * 0xBF58476D1CE4E5B9n) & M64;
      x = ((x ^ (x >> 27n)) * 0x94D049BB133111EBn) & M64;
      this.s[i] = x ^ (x >> 31n);
    }
  }
  nextU64() {
    const s = this.s;
    const result = (rotl((s[1] * 5n) & M64, 7n) * 9n) & M64;
    const t = (s[1] << 17n) & M64;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45n);
    return result;
  }
  uniform() { return (Number(this.nextU64() >> 11n) + 0.5) * (1 / 9007199254740992); }
}

/** bin edges of spectra.rs::bin_edges for L = 1: kmin = 0.9 kf, kmax = 1.01 k_Nyq sqrt(d). */
export function binEdges(dim, n, nbins) {
  const kf = 2 * Math.PI, kmin = kf * 0.9, kmax = Math.PI * n * Math.sqrt(dim) * 1.01;
  return Array.from({ length: nbins + 1 }, (_, i) => kmin * Math.pow(kmax / kmin, i / nbins));
}

/**
 * Lattice modes per |k| bin: shells with few modes are enumerated completely, the others are filled with `perBin` random integer vectors
 * (|m_a| <= n/2) of the bin.  Returns {edges, nbins, nm, modes: Float32Array(4 nm) of [mx, my, mz, bin], kk: Float64Array(nm) |k|,
 * bin: Uint16Array(nm), enumerated: bool[nbins]}.
 */
export function sampleModes(dim, n, nbins, perBin, seed = 1) {
  const edges = binEdges(dim, n, nbins);
  const kf = 2 * Math.PI;
  const rng = new Rng(seed >>> 0);
  const ms = [], kks = [], bins = [], enumerated = new Array(nbins).fill(false);
  for (let b = 0; b < nbins; b++) {
    const klo = edges[b], khi = edges[b + 1];
    const mmax = Math.ceil(khi / kf);
    const countEst = dim === 2 ? Math.PI * (khi * khi - klo * klo) / (kf * kf) : 4 / 3 * Math.PI * (khi ** 3 - klo ** 3) / kf ** 3;
    if (countEst <= perBin * 1.5) {
      enumerated[b] = true;
      const r = Math.min(mmax, n >> 1);
      for (let mx = -r; mx <= r; mx++) for (let my = -r; my <= r; my++) {
        const mzs = dim === 3 ? r : 0;
        for (let mz = -mzs; mz <= mzs; mz++) {
          const k0 = mx * kf, k1 = my * kf, k2 = mz * kf;
          const kk = Math.sqrt(k0 * k0 + k1 * k1 + k2 * k2);
          if (kk >= klo && kk < khi && kk > 0) { ms.push(mx, my, mz); kks.push(kk); bins.push(b); }
        }
      }
    } else {
      let got = 0, tries = 0;
      const span = 2 * mmax + 1;
      while (got < perBin && tries < perBin * 200) {
        tries++;
        const mx = Math.floor(rng.uniform() * span) - mmax;
        const my = Math.floor(rng.uniform() * span) - mmax;
        const mz = dim === 3 ? Math.floor(rng.uniform() * span) - mmax : 0;
        if (Math.abs(mx) > n / 2 || Math.abs(my) > n / 2 || Math.abs(mz) > n / 2) continue;
        const k0 = mx * kf, k1 = my * kf, k2 = mz * kf;
        const kk = Math.sqrt(k0 * k0 + k1 * k1 + k2 * k2);
        if (kk >= klo && kk < khi && kk > 0) { ms.push(mx, my, mz); kks.push(kk); bins.push(b); got++; }
      }
    }
  }
  const nm = kks.length;
  const modes = new Float32Array(4 * nm);
  for (let i = 0; i < nm; i++) { modes[4 * i] = ms[3 * i]; modes[4 * i + 1] = ms[3 * i + 1]; modes[4 * i + 2] = ms[3 * i + 2]; modes[4 * i + 3] = bins[i]; }
  return { edges, nbins, nm, modes, kk: Float64Array.from(kks), bin: Uint16Array.from(bins), enumerated };
}

/** Bin the per-mode transform (re, im interleaved) like direct_spectrum: mean |delta^|^2 / L^d and mean |k| per non-empty bin. */
export function binTransform(sm, dh) {
  const nb = sm.nbins, ks = new Float64Array(nb), ps = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let i = 0; i < sm.nm; i++) {
    const b = sm.bin[i], re = dh[2 * i], im = dh[2 * i + 1];
    ks[b] += sm.kk[i]; ps[b] += re * re + im * im; cnt[b] += 1;
  }
  const k = [], p = [], c = [], bin = [];
  for (let b = 0; b < nb; b++) if (cnt[b] > 0) { k.push(ks[b] / cnt[b]); p.push(ps[b] / cnt[b]); c.push(cnt[b]); bin.push(b); }
  return { k: Float64Array.from(k), p: Float64Array.from(p), n: Float64Array.from(c), bin: Uint16Array.from(bin) };
}

// ---------------------------------------------------------------------------------------------------------------------------------
// f64 reference (port of exp_divided_difference / simplex_transform / direct_spectrum's simplex loop); small problems only

const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
const cexp = (a) => { const e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; };
const cabs = (a) => Math.hypot(a[0], a[1]);

/** Divided difference of exp at complex nodes `s` ([re, im] pairs), stable for coincident nodes (core/src/direct.rs). */
export function expDividedDifference(s) {
  const n = s.length;
  if (n === 1) return cexp(s[0]);
  let dmin = Infinity;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dmin = Math.min(dmin, cabs(csub(s[i], s[j])));
  if (dmin > 0.05) {
    const t = s.map(cexp);
    for (let level = 1; level < n; level++) for (let i = 0; i < n - level; i++) t[i] = cdiv(csub(t[i + 1], t[i]), csub(s[i + level], s[i]));
    return t[0];
  }
  let mean = [0, 0];
  for (const v of s) mean = cadd(mean, v);
  mean = [mean[0] / n, mean[1] / n];
  const d = s.map((v) => csub(v, mean));
  const mmax = 24;
  const h = Array.from({ length: mmax + 1 }, () => [0, 0]);
  h[0] = [1, 0];
  for (let k = 0; k < n; k++) for (let m = 1; m <= mmax; m++) h[m] = cadd(h[m], cmul(d[k], h[m - 1]));
  let fact = 1;
  for (let i = 1; i < n; i++) fact *= i;
  let sum = [0, 0], f = fact;
  for (let m = 0; m <= mmax; m++) {
    if (m > 0) f *= (m + n - 1);
    const term = [h[m][0] / f, h[m][1] / f];
    sum = cadd(sum, term);
    if (cabs(term) < 1e-18 * Math.max(cabs(sum), 1e-300) && m > 2) break;
  }
  return cmul(cexp(mean), sum);
}

/** sum over the simplices of dx^d * (P0: exp[s]; P1: sum_j w'_j exp[s, s_j]) at the integer mode m (the L = 1 lattice mode k = 2 pi m). */
export function directTransformRef(dim, n, pos, w, p1, modes) {
  const nm = modes.length / 4, d = dim, kf = 2 * Math.PI;
  const out = new Float64Array(2 * nm);
  const dxd = Math.pow(1 / n, d);
  const get = (ijk) => {
    const idx = [0, 0, 0], sh = [0, 0, 0];
    for (let a = 0; a < d; a++) { if (ijk[a] >= n) { idx[a] = ijk[a] - n; sh[a] = 1; } else idx[a] = ijk[a]; }
    const flat = d === 2 ? idx[0] * n + idx[1] : (idx[0] * n + idx[1]) * n + idx[2];
    const p = [0, 0, 0];
    for (let a = 0; a < d; a++) p[a] = pos[flat * d + a] + sh[a];
    return { p, w: w ? w[flat] : 1 };
  };
  const TETS = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
  const proc = (vs) => {
    const nt = vs.length;
    let wm = 0;
    for (const v of vs) wm += v.w;
    wm /= nt;
    if (!(wm > 0)) wm = 1;
    for (let mi = 0; mi < nm; mi++) {
      const k = [modes[4 * mi] * kf, modes[4 * mi + 1] * kf, modes[4 * mi + 2] * kf];
      const s = vs.map((v) => [0, -(k[0] * v.p[0] + k[1] * v.p[1] + k[2] * v.p[2])]);
      let tot;
      if (!p1) tot = expDividedDifference(s);
      else {
        tot = [0, 0];
        for (let j = 0; j < nt; j++) tot = cadd(tot, cmul(expDividedDifference([...s, s[j]]), [vs[j].w / wm, 0]));
      }
      out[2 * mi] += tot[0] * dxd; out[2 * mi + 1] += tot[1] * dxd;
    }
  };
  if (d === 2) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const c = [get([i, j, 0]), get([i + 1, j, 0]), get([i, j + 1, 0]), get([i + 1, j + 1, 0])];
      proc([c[0], c[1], c[3]]); proc([c[0], c[3], c[2]]);
    }
  } else {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
      const c = [];
      for (let b = 0; b < 8; b++) c.push(get([i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)]));
      for (const t of TETS) proc([c[t[0]], c[t[1]], c[t[2]], c[t[3]]]);
    }
  }
  return out;
}

/** Gauss-Legendre nodes and weights on [0, 1]. */
function gaussLegendre01(m) {
  const x = new Float64Array(m), w = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (m + 0.5)), pp = 1;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 0; j < m; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1); }
      pp = m * (z * p1 - p2) / (z * z - 1);
      const dz = p1 / pp; z -= dz;
      if (Math.abs(dz) < 1e-15) break;
    }
    x[i] = 0.5 * (1 - z); w[i] = 1 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
}

/**
 * Independent reference: the same sum evaluated by tensor Gauss-Legendre quadrature on the collapsed (Duffy) simplex, i.e. the
 * Hermite-Genocchi integral itself, int_Delta wlin(lambda) exp(-i k.x(lambda)) d lambda.  No divided differences; slow (ngl^d points per
 * simplex and mode), for tiny meshes only.  ngl must resolve the phase spread of a simplex (about spread/2 + 10 points).
 */
export function directTransformQuad(dim, n, pos, w, p1, modes, ngl = 24) {
  const nm = modes.length / 4, d = dim, kf = 2 * Math.PI;
  const out = new Float64Array(2 * nm);
  const dxd = Math.pow(1 / n, d);
  const { x: gx, w: gw } = gaussLegendre01(ngl);
  // quadrature points on the standard simplex: barycentric coordinates lambda[0..d] and weights (sum = 1/d!)
  const lam = [], om = [];
  if (d === 2) {
    for (let a = 0; a < ngl; a++) for (let b = 0; b < ngl; b++) {
      const u = gx[a], v = gx[b];
      const x1 = u, x2 = v * (1 - u);
      lam.push([1 - x1 - x2, x1, x2]); om.push(gw[a] * gw[b] * (1 - u));
    }
  } else {
    for (let a = 0; a < ngl; a++) for (let b = 0; b < ngl; b++) for (let c = 0; c < ngl; c++) {
      const u = gx[a], v = gx[b], t = gx[c];
      const x1 = u, x2 = v * (1 - u), x3 = t * (1 - u) * (1 - v);
      lam.push([1 - x1 - x2 - x3, x1, x2, x3]); om.push(gw[a] * gw[b] * gw[c] * (1 - u) * (1 - u) * (1 - v));
    }
  }
  const np = lam.length;
  const get = (ijk) => {
    const idx = [0, 0, 0], sh = [0, 0, 0];
    for (let a = 0; a < d; a++) { if (ijk[a] >= n) { idx[a] = ijk[a] - n; sh[a] = 1; } else idx[a] = ijk[a]; }
    const flat = d === 2 ? idx[0] * n + idx[1] : (idx[0] * n + idx[1]) * n + idx[2];
    const p = [0, 0, 0];
    for (let a = 0; a < d; a++) p[a] = pos[flat * d + a] + sh[a];
    return { p, w: w ? w[flat] : 1 };
  };
  const TETS = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
  const proc = (vs) => {
    const nt = vs.length;
    let wm = 0;
    for (const v of vs) wm += v.w;
    wm /= nt;
    if (!(wm > 0)) wm = 1;
    for (let mi = 0; mi < nm; mi++) {
      const k0 = modes[4 * mi] * kf, k1 = modes[4 * mi + 1] * kf, k2 = modes[4 * mi + 2] * kf;
      const ph = vs.map((v) => -(k0 * v.p[0] + k1 * v.p[1] + k2 * v.p[2]));
      let re = 0, im = 0;
      for (let q = 0; q < np; q++) {
        const L = lam[q];
        let a = 0, wl = 0;
        for (let j = 0; j < nt; j++) { a += L[j] * ph[j]; wl += L[j] * vs[j].w; }
        const f = om[q] * (p1 ? wl / wm : 1);
        re += f * Math.cos(a); im += f * Math.sin(a);
      }
      out[2 * mi] += re * dxd; out[2 * mi + 1] += im * dxd;
    }
  };
  if (d === 2) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const c = [get([i, j, 0]), get([i + 1, j, 0]), get([i, j + 1, 0]), get([i + 1, j + 1, 0])];
      proc([c[0], c[1], c[3]]); proc([c[0], c[3], c[2]]);
    }
  } else {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
      const c = [];
      for (let b = 0; b < 8; b++) c.push(get([i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)]));
      for (const t of TETS) proc([c[t[0]], c[t[1]], c[t[2]], c[t[3]]]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// WGSL

const wgslMain = (dim, p1) => {
  const nv = 1 << dim, ns = dim === 2 ? 2 : 6, nt = dim + 1, nn = p1 ? nt + 1 : nt;
  const simp = dim === 2 ? [0, 1, 3, 0, 3, 2] : [0, 1, 3, 7, 0, 1, 5, 7, 0, 2, 3, 7, 0, 2, 6, 7, 0, 4, 5, 7, 0, 4, 6, 7];
  return /* wgsl */`
${WGSL_LINEAR}
const DIM: u32 = ${dim}u;
const NV: u32 = ${nv}u;
const NS: u32 = ${ns}u;
const NT: u32 = ${nt}u;
const NN: u32 = ${nn}u;
const TWO_PI: f32 = 6.283185307179586;
const THR: f32 = ${DIRECT_THR.toFixed(3)};

struct DU { nm: u32, n: u32, cpc: u32, ncell: u32, off0: u32, off1: u32, ctotal: u32, useW: u32, invn: f32, dxd: f32 };
@group(0) @binding(0) var<uniform> du: DU;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> wts: array<f32>;
@group(0) @binding(3) var<storage, read> modes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> partial: array<vec2<f32>>;

var<private> SIMP: array<u32, ${ns * nt}> = array<u32, ${ns * nt}>(${simp.map((v) => v + 'u').join(', ')});
var<private> PT: array<f32, 5>;               // node turns t_j = m.(v_j - v_0) of the current divided difference, sorted ascending
var<private> FF: array<vec2<f32>, 15>;        // divided-difference table, level l at offset l*NN - l(l-1)/2
var<private> HH: array<f32, 24>;

fn cph(turns: f32) -> vec2<f32> {              // exp(-2 pi i turns)
  let a = -TWO_PI * fract(turns);
  return vec2<f32>(cos(a), sin(a));
}
fn cm(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// exp[x_lo, .., x_{lo+cnt-1}], x_j = -2 pi i PT[j], for nodes whose span is below THR: Taylor series about the mean node,
//   e^{xbar} sum_m i^m h_m(psi) / (m + cnt - 1)!,   psi_j = -2 pi (PT[j] - tbar),
// h_m the complete homogeneous symmetric polynomials (nodes are imaginary, so h_m(i psi) = i^m h_m(psi): the recursion is real)
fn dd_taylor(lo: u32, cnt: u32) -> vec2<f32> {
  var tsum = 0.0;
  for (var a = 0u; a < cnt; a++) { tsum += PT[lo + a]; }
  let tb = tsum / f32(cnt);
  var R = 0.0;
  for (var a = 0u; a < cnt; a++) { R += abs(PT[lo + a] - tb); }
  R = R * TWO_PI;
  let l = cnt - 1u;
  var f0 = 1.0;
  for (var q = 2u; q <= l; q++) { f0 = f0 / f32(q); }
  var M = 1u;
  var bnd = f0;
  loop {
    bnd = bnd * R / f32(M + l);
    if (bnd < 1.0e-8 * f0 || M >= 22u) { break; }
    M++;
  }
  HH[0] = 1.0;
  for (var m = 1u; m <= M; m++) { HH[m] = 0.0; }
  for (var a = 0u; a < cnt; a++) {
    let psi = -TWO_PI * (PT[lo + a] - tb);
    for (var m = 1u; m <= M; m++) { HH[m] = HH[m] + psi * HH[m - 1u]; }
  }
  var S = vec2<f32>(0.0);
  var cmf = f0;
  for (var m = 0u; m <= M; m++) {
    if (m > 0u) { cmf = cmf / f32(m + l); }
    let t = HH[m] * cmf;
    let r = m & 3u;
    if (r == 0u) { S.x += t; } else if (r == 1u) { S.y += t; } else if (r == 2u) { S.x -= t; } else { S.y -= t; }
  }
  return cm(S, cph(tb));
}

// exp[x_0, .., x_{NN-1}] * exp(-2 pi i T0) for the nodes in PT[0..NN) (T0: turns of the base vertex).  Sorted nodes; Newton table
//   f[i..i+l] = (f[i+1..i+l] - f[i..i+l-1]) / (x_{i+l} - x_i)    where the span of the entry is at least THR,
// the local Taylor series (above) for every entry whose nodes lie within THR (clusters, coincident nodes, the repeated P1 node).
fn dd_eval(T0: f32) -> vec2<f32> {
  for (var i = 1u; i < NN; i++) {
    let v = PT[i];
    var j = i;
    loop {
      if (j == 0u) { break; }
      if (PT[j - 1u] <= v) { break; }
      PT[j] = PT[j - 1u];
      j = j - 1u;
    }
    PT[j] = v;
  }
  if (TWO_PI * (PT[NN - 1u] - PT[0]) < THR) { return cm(dd_taylor(0u, NN), cph(T0)); }
  for (var k = 0u; k < NN; k++) { FF[k] = cph(PT[k]); }
  var op = 0u;
  var o = NN;
  for (var l = 1u; l < NN; l++) {
    for (var i = 0u; i < NN - l; i++) {
      let sp = TWO_PI * (PT[i + l] - PT[i]);
      if (sp < THR) {
        if (l == 1u) {
          let x = 0.5 * sp;
          var sc = 1.0 - x * x / 6.0;
          if (x > 1.0e-3) { sc = sin(x) / x; }
          FF[o + i] = cph(0.5 * (PT[i] + PT[i + 1u])) * sc;
        } else {
          FF[o + i] = dd_taylor(i, l + 1u);
        }
      } else {
        let df = FF[op + i + 1u] - FF[op + i];
        FF[o + i] = vec2<f32>(-df.y, df.x) / sp;           // division by x_{i+l} - x_i = -i sp
      }
    }
    op = o;
    o = o + NN - l;
  }
  return cm(FF[op], cph(T0));
}

${p1 && dim === 3 ? `
var<private> PTK: array<f32, 4>;              // P1: node turns of the simplex's own vertices (unsorted) and their normalised weights
var<private> WK: array<f32, 4>;
var<private> BT: array<f32, 4>;               // the same nodes sorted ascending, BP = original index
var<private> BP: array<u32, 4>;
var<private> BF: array<vec2<f32>, 16>;        // divided differences of the NT distinct nodes, [a * NT + b]
var<private> NW: array<vec2<f32>, 25>;        // the entries of the (NT + 1)-node set with the node q repeated that contain both copies, [a * (NT + 1) + b]

fn mp(s: u32, q: u32) -> u32 { return select(s - 1u, s, s <= q); }       // position in the repeated sequence -> position among the NT nodes
fn ent(a: u32, b: u32, q: u32) -> vec2<f32> {
  if (a <= q && b >= q + 1u) { return NW[a * (NT + 1u) + b]; }
  return BF[mp(a, q) * NT + mp(b, q)];
}

// sum_j WK[j] exp[x_0, .., x_{NT-1}, x_j] * exp(-2 pi i T0): the table of the distinct nodes is built once; for each repeated node q only the
// entries that contain both copies are new (the others are entries of the shared table)
fn p1_eval(T0: f32) -> vec2<f32> {
  for (var k = 0u; k < NT; k++) { BT[k] = PTK[k]; BP[k] = k; }
  for (var i = 1u; i < NT; i++) {
    let v = BT[i]; let pidx = BP[i];
    var j = i;
    loop {
      if (j == 0u) { break; }
      if (BT[j - 1u] <= v) { break; }
      BT[j] = BT[j - 1u]; BP[j] = BP[j - 1u];
      j = j - 1u;
    }
    BT[j] = v; BP[j] = pidx;
  }
  var acc = vec2<f32>(0.0);
  if (TWO_PI * (BT[NT - 1u] - BT[0]) < THR) {
    for (var q = 0u; q < NT; q++) {
      for (var s = 0u; s <= NT; s++) { PT[s] = BT[select(s - 1u, s, s <= q)]; }
      acc += WK[BP[q]] * dd_taylor(0u, NN);
    }
    return cm(acc, cph(T0));
  }
  for (var k = 0u; k < NT; k++) { PT[k] = BT[k]; BF[k * NT + k] = cph(BT[k]); }
  for (var l = 1u; l < NT; l++) {
    for (var a = 0u; a + l < NT; a++) {
      let b = a + l;
      let sp = TWO_PI * (BT[b] - BT[a]);
      if (sp < THR) {
        if (l == 1u) {
          let x = 0.5 * sp;
          var sc = 1.0 - x * x / 6.0;
          if (x > 1.0e-3) { sc = sin(x) / x; }
          BF[a * NT + b] = cph(0.5 * (BT[a] + BT[b])) * sc;
        } else { BF[a * NT + b] = dd_taylor(a, l + 1u); }
      } else {
        let df = BF[(a + 1u) * NT + b] - BF[a * NT + b - 1u];
        BF[a * NT + b] = vec2<f32>(-df.y, df.x) / sp;
      }
    }
  }
  for (var q = 0u; q < NT; q++) {
    for (var s = 0u; s <= NT; s++) { PT[s] = BT[select(s - 1u, s, s <= q)]; }
    for (var l = 1u; l <= NT; l++) {
      for (var a = 0u; a + l <= NT; a++) {
        let b = a + l;
        if (a > q || b < q + 1u) { continue; }
        var v = vec2<f32>(0.0);
        if (l == 1u) { v = BF[q * NT + q]; }
        else {
          let sp = TWO_PI * (PT[b] - PT[a]);
          if (sp < THR) { v = dd_taylor(a, l + 1u); }
          else {
            let df = ent(a + 1u, b, q) - ent(a, b - 1u, q);
            v = vec2<f32>(-df.y, df.x) / sp;
          }
        }
        NW[a * (NT + 1u) + b] = v;
      }
    }
    acc += WK[BP[q]] * NW[NT];
  }
  return cm(acc, cph(T0));
}` : ''}

@compute @workgroup_size(64)
fn dd_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let id = linear_id(gid, nwg, 64u);
  if (id >= du.nm * du.ctotal) { return; }
  let mi = id % du.nm;
  let chunk = id / du.nm;
  let md = modes[mi];
  let m3 = vec3<i32>(i32(md.x), i32(md.y), i32(md.z));
  let mf = vec3<f32>(md.x, md.y, md.z);
  let n = du.n;
  let ni = i32(n);
  // this thread's chunk is the cells [chunk cpc, (chunk + 1) cpc); a submit handles the part [off0, off1) of every chunk
  let cb = chunk * du.cpc;
  let c0 = cb + du.off0;
  let c1 = min(min(cb + du.off1, cb + du.cpc), du.ncell);
  var acc = vec2<f32>(0.0);
  for (var cell = c0; cell < c1; cell++) {
    var i0 = 0u; var j0 = 0u; var k0 = 0u;
    if (DIM == 2u) { i0 = cell / n; j0 = cell % n; } else { i0 = cell / (n * n); j0 = (cell / n) % n; k0 = cell % n; }
    var psi: array<vec3<f32>, NV>;
    var cw: array<f32, NV>;
    for (var b = 0u; b < NV; b++) {
      var ix = i0 + (b & 1u); if (ix >= n) { ix -= n; }
      var jy = j0 + ((b >> 1u) & 1u); if (jy >= n) { jy -= n; }
      var kz = 0u;
      var flat = ix * n + jy;
      if (DIM == 3u) { kz = k0 + ((b >> 2u) & 1u); if (kz >= n) { kz -= n; } flat = flat * n + kz; }
      let o = flat * DIM;
      var p = vec3<f32>(pos[o] - f32(ix) * du.invn, pos[o + 1u] - f32(jy) * du.invn, 0.0);
      if (DIM == 3u) { p.z = pos[o + 2u] - f32(kz) * du.invn; }
      psi[b] = p;
      cw[b] = select(1.0, wts[flat], du.useW != 0u);
    }
    for (var s = 0u; s < NS; s++) {
      var vb: array<u32, NT>;
      for (var k = 0u; k < NT; k++) { vb[k] = SIMP[s * NT + k]; }
      let b0 = vb[0];
      // turns of the base vertex: exact lattice part ((m.L) mod n)/n plus m.Psi
      let L0 = vec3<i32>(i32(i0 + (b0 & 1u)), i32(j0 + ((b0 >> 1u) & 1u)), i32(k0 + ((b0 >> 2u) & 1u)));
      var r = (m3.x * L0.x + m3.y * L0.y + m3.z * L0.z) % ni;
      if (r < 0) { r += ni; }
      let T0 = fract(f32(r) * du.invn + dot(mf, psi[b0]));
      var tk: array<f32, NT>;
      var wk: array<f32, NT>;
      var wsum = 0.0;
      for (var k = 0u; k < NT; k++) {
        let bk = vb[k];
        let db = vec3<i32>(i32(bk & 1u) - i32(b0 & 1u), i32((bk >> 1u) & 1u) - i32((b0 >> 1u) & 1u), i32((bk >> 2u) & 1u) - i32((b0 >> 2u) & 1u));
        let mdb = m3.x * db.x + m3.y * db.y + m3.z * db.z;
        tk[k] = f32(mdb) * du.invn + dot(mf, psi[bk] - psi[b0]);
        wk[k] = cw[bk]; wsum += cw[bk];
      }
${p1 ? `
      var wmean = wsum / f32(NT);
      if (!(wmean > 0.0)) { wmean = 1.0; for (var k = 0u; k < NT; k++) { wk[k] = 1.0; } }
${dim === 3 ? `      for (var k = 0u; k < NT; k++) { PTK[k] = tk[k]; WK[k] = wk[k] / wmean; }
      let v = p1_eval(T0);
      if (abs(v.x) < 4.0 && abs(v.y) < 4.0) { acc += v; }` : `      for (var jj = 0u; jj < NT; jj++) {
        for (var k = 0u; k < NT; k++) { PT[k] = tk[k]; }
        PT[NT] = tk[jj];
        let v = dd_eval(T0);
        if (abs(v.x) < 1.0 && abs(v.y) < 1.0) { acc += (wk[jj] / wmean) * v; }
      }`}` : `
      for (var k = 0u; k < NT; k++) { PT[k] = tk[k]; }
      let v = dd_eval(T0);
      if (abs(v.x) < 1.0 && abs(v.y) < 1.0) { acc += v; }`}
    }
  }
  let pidx = mi * du.ctotal + chunk;
  partial[pidx] = partial[pidx] + acc * du.dxd;
}
`;
};

const WGSL_REDUCE = /* wgsl */`
struct RU2 { nm: u32, ctotal: u32 };
@group(0) @binding(0) var<uniform> ru: RU2;
@group(0) @binding(1) var<storage, read> part: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> res: array<vec2<f32>>;
var<workgroup> sh: array<vec2<f32>, 64>;
@compute @workgroup_size(64)
fn dd_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let mi = wid.x;
  var s = vec2<f32>(0.0);
  var c = lid;
  loop {
    if (c >= ru.ctotal) { break; }
    s += part[mi * ru.ctotal + c];
    c += 64u;
  }
  sh[lid] = s;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st = st >> 1u) {
    if (lid < st) { sh[lid] += sh[lid + st]; }
    workgroupBarrier();
  }
  if (lid == 0u) { res[mi] = sh[0]; }
}
`;

const pipeCache = new WeakMap();
function getPipes(device) {
  let p = pipeCache.get(device);
  if (!p) { p = { main: new Map(), reduce: null }; pipeCache.set(device, p); }
  return p;
}

/** Work per submit (simplex x mode evaluations) before the dispatch is split (keeps one command buffer well under the GPU watchdog). */
const EVALS_PER_SUBMIT = 1.2e9;

export class GpuDirectSpectrum {
  constructor(device) {
    this.device = device; this.ring = ParamRing.get(device);
    this.dummy = null; this.destroyed = false;
    this.last = null;
  }
  async init() {
    const P = getPipes(this.device);
    if (!P.reduce) {
      const m = await compile(this.device, WGSL_REDUCE, 'direct reduce');
      P.reduce = makePipeline(this.device, m, 'dd_reduce', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'dd_reduce');
    }
    this.dummy = this.device.createBuffer({ label: 'direct no weights', size: 16, usage: STORAGE_RW });
    return this;
  }
  destroy() {
    this.destroyed = true;
    for (const b of [this.dummy, this.partial, this.modesBuf, this.resBuf]) if (b) b.destroy();
    this.dummy = this.partial = this.modesBuf = this.resBuf = null;
  }
  async mainPipe(dim, p1) {
    const P = getPipes(this.device), key = dim + (p1 ? 'p' : 'c');
    let e = P.main.get(key);
    if (!e) {
      e = (async () => {
        const m = await compile(this.device, wgslMain(dim, p1), `direct d${dim}${p1 ? ' P1' : ' P0'}`);
        return makePipeline(this.device, m, 'dd_main', [U_ENTRY(0), RO_ENTRY(1), RO_ENTRY(2), RO_ENTRY(3), RW_ENTRY(4)], `dd_main ${key}`);
      })();
      P.main.set(key, e);
    }
    return e;
  }

  /**
   * Per-mode complex transform delta^(k = 2 pi m) = sum over the simplices; `modes`: Float32Array of [mx, my, mz, bin] quadruples.
   * Returns {delta: Float32Array(2 nm) (re, im), ms, chunks, batches}.
   */
  async transform({ dim, n, pos, w = null, p1 = false, modes }) {
    const dev = this.device, nm = modes.length / 4;
    if (nm === 0) return { delta: new Float32Array(0), ms: 0, chunks: 0, batches: 0 };
    if (!Number.isInteger(Math.log2(n))) throw new Error('GpuDirectSpectrum: n must be a power of two');
    if (p1 && !w) throw new Error('GpuDirectSpectrum: P1 needs the vertex weights');
    const t0 = performance.now();
    const pipe = await this.mainPipe(dim, p1);
    const ncell = n ** dim, ns = dim === 2 ? 2 : 6;
    // chunks per mode: ~2^19 threads in total, at least ~8 cells per chunk.  Large problems are split into `batches` submits, each handling a
    // slice of the cells of EVERY chunk (so every submit keeps the whole GPU busy and stays below EVALS_PER_SUBMIT evaluations); the chunk
    // sums accumulate in the partials buffer.
    let ctotal = 1;
    while (ctotal * nm < (1 << 19) && ctotal * 2 * 8 <= ncell) ctotal *= 2;
    const evals = ncell * ns * nm * (p1 ? dim + 1 : 1);
    const cpc = Math.ceil(ncell / ctotal);
    let batches = 1;
    while (batches * 2 <= cpc && evals / batches > EVALS_PER_SUBMIT) batches *= 2;
    const sub = Math.ceil(cpc / batches);
    if (this.partial && this.partial.size < nm * ctotal * 8) { this.partial.destroy(); this.partial = null; }
    if (!this.partial) this.partial = dev.createBuffer({ label: 'direct partials', size: Math.max(16, nm * ctotal * 8), usage: STORAGE_RW });
    if (this.modesBuf && this.modesBuf.size < modes.byteLength) { this.modesBuf.destroy(); this.modesBuf = null; }
    if (!this.modesBuf) this.modesBuf = dev.createBuffer({ label: 'direct modes', size: Math.max(16, modes.byteLength), usage: STORAGE_RW });
    dev.queue.writeBuffer(this.modesBuf, 0, modes);
    if (this.resBuf && this.resBuf.size < nm * 8) { this.resBuf.destroy(); this.resBuf = null; }
    if (!this.resBuf) this.resBuf = dev.createBuffer({ label: 'direct result', size: Math.max(16, nm * 8), usage: STORAGE_RW });
    const invn = 1 / n, dxd = Math.pow(invn, dim);
    for (let b = 0; b < batches; b++) {
      const slot = this.ring.write([['u', nm], ['u', n], ['u', cpc], ['u', ncell], ['u', b * sub], ['u', Math.min((b + 1) * sub, cpc)], ['u', ctotal], ['u', w ? 1 : 0], invn, dxd]);
      const bg = makeBindGroup(dev, pipe.bgl, [this.ring.resource(48), pos, w || this.dummy, this.modesBuf, this.partial], 'direct bg');
      const enc = dev.createCommandEncoder({ label: 'direct stage 1' });
      if (b === 0) enc.clearBuffer(this.partial, 0, nm * ctotal * 8);
      const pass = enc.beginComputePass({ label: 'direct stage 1' });
      pass.setPipeline(pipe.pipeline);
      pass.setBindGroup(0, bg, [slot]);
      const groups = Math.ceil(nm * ctotal / 64);
      if (groups <= 65535) pass.dispatchWorkgroups(groups, 1, 1); else pass.dispatchWorkgroups(32768, Math.ceil(groups / 32768), 1);
      pass.end();
      dev.queue.submit([enc.finish()]);
      if (batches > 1) await dev.queue.onSubmittedWorkDone();
    }
    {
      const R = getPipes(dev).reduce;
      const slot = this.ring.write([['u', nm], ['u', ctotal]]);
      const bg = makeBindGroup(dev, R.bgl, [this.ring.resource(16), this.partial, this.resBuf], 'direct reduce bg');
      const enc = dev.createCommandEncoder({ label: 'direct stage 2' });
      const pass = enc.beginComputePass({ label: 'direct stage 2' });
      pass.setPipeline(R.pipeline);
      pass.setBindGroup(0, bg, [slot]);
      pass.dispatchWorkgroups(nm);
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    const delta = await readF32(dev, this.resBuf, 2 * nm);
    const ms = performance.now() - t0;
    this.last = { nm, ctotal, batches, cpc, ms };
    return { delta, ms, chunks: ctotal, batches };
  }

  /** Sample the modes (see sampleModes), run the kernel and bin: {k, p, n, nmodes, ms, sm}. */
  async spectrum({ dim, n, pos, w = null, p1 = false, nbins, perBin = 128, seed = 1, sm = null }) {
    sm = sm || sampleModes(dim, n, nbins, perBin, seed);
    const r = await this.transform({ dim, n, pos, w, p1, modes: sm.modes });
    const s = binTransform(sm, r.delta);
    return { k: s.k, p: s.p, n: s.n, nmodes: sm.nm, ms: r.ms, chunks: r.chunks, batches: r.batches, sm };
  }
}
