// engine.js - owns the CosmoSim (WASM) instance and every cached derived quantity.
//
// All getters take the *applied* parameter snapshot P (see app.js) and are
// memoised on (IC generation, relevant parameters). No DOM access here.

import { unpackSpectrum, unpackLoop, fmap, transpose2D } from '../hcc.js';

const MAX_CACHE_BYTES = 420e6;

function sizeOf(v) {
  if (v == null) return 0;
  if (ArrayBuffer.isView(v)) return v.byteLength;
  if (typeof v === 'object') { let s = 0; for (const k in v) s += sizeOf(v[k]); return s; }
  return 8;
}

export const NBINS = { 2: 48, 3: 40 };

export class Engine {
  constructor(core) {
    this.core = core;
    this.sim = null;
    this.sig = '';
    this.gen = 0;
    this.builtOrder = 0;
    this.cache = new Map();
    this.cacheBytes = 0;
    this.timings = new Map();
    this.dim = 2; this.n = 0;
    this.gaussian = false;
    this.kf = 2 * Math.PI; this.knyq = 0;
  }

  // -- bookkeeping -------------------------------------------------------------

  time(label, fn) {
    const t0 = performance.now();
    try { return fn(); } finally { this.timings.set(label, performance.now() - t0); }
  }

  dispose() {
    if (this.sim) { try { this.sim.free(); } catch (e) { /* already freed */ } }
    this.sim = null; this.builtOrder = 0;
    this.cache.clear(); this.cacheBytes = 0;
  }

  memo(parts, fn) {
    const key = this.gen + '|' + parts.join('|');
    if (this.cache.has(key)) {
      const v = this.cache.get(key);
      this.cache.delete(key); this.cache.set(key, v);   // LRU touch
      return v.value;
    }
    const value = fn();
    const bytes = sizeOf(value);
    this.cache.set(key, { value, bytes });
    this.cacheBytes += bytes;
    for (const [k, v] of this.cache) {
      if (this.cacheBytes <= MAX_CACHE_BYTES || this.cache.size <= 2) break;
      if (k === key) continue;
      this.cache.delete(k); this.cacheBytes -= v.bytes;
    }
    return value;
  }

  /** Cached value without computing (undefined if absent). */
  peek(parts) {
    const v = this.cache.get(this.gen + '|' + parts.join('|'));
    return v ? v.value : undefined;
  }
  hcCached(P) { return this.peek(['hc', P.D, P.nu, P.me, P.mx]); }

  static icSignature(P) {
    const a = [P.mode, P.n, P.ic, P.R];
    if (P.ic === 'g') a.push(P.shape, P.pn, P.ns, P.gm, P.seed, P.sg);
    else if (P.ic === 'w') a.push(P.k1.join(','), P.k2.join(','), P.a1, P.a2, P.f1, P.f2, P.sp);
    else a.push(P.pa, P.pw, P.sp);
    return a.join('|');
  }

  /** (Re)build the simulation for the IC part of P. Returns true if it was rebuilt. */
  configure(P) {
    const sig = Engine.icSignature(P);
    if (this.sim && sig === this.sig) return false;
    this.dispose();
    const dim = P.mode, n = P.n;
    this.sim = this.time('IC', () => {
      const sim = new this.core.CosmoSim(dim, n, 1.0);
      if (P.ic === 'g') {
        const shape = { pl: 0, bbks: 1, eh: 2 }[P.shape];
        const p1 = P.shape === 'pl' ? P.pn : P.ns;
        const p2 = P.shape === 'pl' ? 0 : P.gm;
        sim.set_ic_gaussian(shape, p1, p2, P.R, P.seed >>> 0, P.sg);
      } else if (P.ic === 'w') {
        const modes = [];
        const add = (k, a, f) => {
          const kz = dim === 3 ? k[2] : 0;
          if ((k[0] || k[1] || kz) && a > 0) modes.push(k[0], k[1], kz, a, f * Math.PI / 180);
        };
        add(P.k1, P.a1, P.f1); add(P.k2, P.a2, P.f2);
        if (!modes.length) modes.push(1, 0, 0, 1, 0);
        sim.set_ic_plane_waves(new Float64Array(modes), P.R, P.sp);
      } else {
        sim.set_ic_peak(P.pa, P.pw, P.R, P.sp);
      }
      return sim;
    });
    this.sig = sig; this.gen++;
    this.dim = dim; this.n = n;
    this.gaussian = P.ic === 'g';
    this.kf = this.sim.kf(); this.knyq = this.sim.knyq();
    this.builtOrder = 0;
    return true;
  }

  ensureLpt(order) {
    if (this.builtOrder >= order) return false;
    this.time('LPT' + order, () => this.sim.build_lpt(order));
    this.builtOrder = order;
    return true;
  }

  // -- fields --------------------------------------------------------------------

  positions(P) { return this.sim.positions(P.D, Math.min(P.order, this.builtOrder)); }

  sheet(P) {
    if (this.dim !== 2) throw new Error('sheet density is only available in 2D');   // a Rust panic would poison the sim
    return this.memo(['sheet', P.D, P.order], () => this.time('sheet', () => this.sim.sheet_density(P.D, P.order, this.n, 2)));
  }
  cic(P) {
    return this.memo(['cic', P.D, P.order], () => this.time('CIC', () => this.sim.cic_density(P.D, P.order, this.n)));
  }
  linear(P) {
    return this.memo(['lin', P.D], () => this.time('linear', () => this.sim.linear_delta(P.D)));
  }
  hc(P) {
    return this.memo(['hc', P.D, P.nu, P.me, P.mx], () => this.time('Hopf-Cole', () => {
      const s = this.sim;
      s.hopf_cole(P.D, P.nu, P.me, P.mx);
      const r = { delta: s.hc_delta(), nuEff: s.hc_nu_eff(), range: s.hc_exponent_range() };
      if (this.dim === 2) {
        r.phi = s.hc_phi(); r.lnpsi = s.hc_lnpsi(); r.psihat = s.hc_psihat_log(); r.vel = s.hc_velocity();
      }
      return r;
    }));
  }
  jacobian(P) {
    return this.memo(['jac', P.D, P.order], () => this.time('Jacobian', () => this.sim.jacobian(P.D, P.order)));
  }
  div(P, nth) {
    this.ensureLpt(Math.max(nth, P.order));
    return this.memo(['div', nth], () => this.sim.lpt_div(nth));
  }
  curl(P, nth) {
    this.ensureLpt(Math.max(nth, P.order));
    return this.memo(['curl', nth], () => this.sim.lpt_curl(nth, 0));
  }
  /** First shell-crossing D for the given order (Infinity if none). */
  dsc(order) {
    this.ensureLpt(order);
    return this.memo(['dsc', order], () => this.time('D_sc' + order, () => this.sim.shell_crossing(order)));
  }

  /** Parameter parts identifying a density-like field. */
  fieldKey(which, P) {
    switch (which) {
      case 'sheet': case 'cic': return [which, P.D, P.order];
      case 'hc': return [which, P.D, P.nu, P.me, P.mx];
      default: return ['lin', P.D];
    }
  }
  /** Overdensity delta (mean 0) of the named field. */
  delta(which, P) {
    return this.memo(['delta', ...this.fieldKey(which, P)], () => {
      if (which === 'hc') return this.hc(P).delta;
      if (which === 'lin') return this.linear(P);
      const rho = which === 'sheet' ? this.sheet(P) : this.cic(P);
      return fmap(rho, (v) => v - 1);
    });
  }
  /** Density 1+delta (3D visualisation); linear theory clipped at 1e-3 so it can be shown on a log scale. */
  rho(which, P) {
    return this.memo(['rho', ...this.fieldKey(which, P)], () => {
      if (which === 'cic') return this.cic(P);
      return fmap(this.delta(which, P), (v) => Math.max(1e-3, v + 1));
    });
  }
  /** Fourier maps: amp = log10|f^|/max, phase; both fft-shifted, as [ikx*n+iky]. */
  fmaps(which, P) {
    return this.memo(['fmap', ...this.fieldKey(which, P)], () => this.time('Fourier maps', () => {
      const d = this.delta(which, P);
      const both = this.sim.fourier_maps(d); // one FFT for both maps
      const nn = both.length / 2;
      return { amp: both.slice(0, nn), phase: both.slice(nn) };
    }));
  }

  // -- spectra ---------------------------------------------------------------------

  get nbins() { return NBINS[this.dim]; }

  /** Binned P(k); CIC is window-deconvolved when deconv is true. */
  pk(which, P, deconv = false) {
    const dc = which === 'cic' && deconv;
    return this.memo(['pk', ...this.fieldKey(which, P), dc], () => this.time('P(k)', () =>
      unpackSpectrum(this.sim.power_spectrum(this.delta(which, P), this.nbins, dc))));
  }
  /** Cross-correlation coefficient r(k) of the named field with the linear field. */
  rk(which, P) {
    return this.memo(['rk', ...this.fieldKey(which, P)], () => this.time('r(k)', () => {
      const dw = this.delta(which, P), dl = this.linear(P);
      const c = unpackSpectrum(this.sim.cross_spectrum(dw, dl, this.nbins));
      const pw = this.pk(which, P, false), pl = this.pk('lin', P, false);
      const r = new Float64Array(c.k.length);
      for (let i = 0; i < r.length; i++) r[i] = c.p[i] / Math.sqrt(Math.max(1e-300, pw.p[i] * pl.p[i]));
      return { k: c.k, r };
    }));
  }
  /** Smooth linear theory D^2 P0(k) on a fine k grid. */
  plinFine(P) {
    return this.memo(['plinfine', P.D], () => {
      const m = 200, k0 = this.kf * 0.8, k1 = this.knyq * Math.sqrt(this.dim);
      const k = new Float64Array(m);
      for (let i = 0; i < m; i++) k[i] = k0 * Math.pow(k1 / k0, i / (m - 1));
      return { k, p: this.sim.linear_pk(k, P.D) };
    });
  }
  /** One-loop terms at the binned k <= k_Nyq. kernels 0 = SPT, 1 = Zel'dovich. */
  loop(P, kernels) {
    if (!this.gaussian) return null;
    return this.memo(['loop', P.D, kernels], () => this.time(kernels ? '1-loop ZA' : '1-loop SPT', () => {
      const ks = this.pk('lin', P, false).k;
      let m = 0; while (m < ks.length && ks[m] <= this.knyq) m++;
      const k = Float64Array.from(ks.subarray(0, m));
      return { k, ...unpackLoop(this.sim.one_loop(k, P.D, kernels)) };
    }));
  }
  /** Per-axis rms Zel'dovich displacement^2 at D = 1 from the discrete mode sum (Gaussian ICs only). */
  sigmaV2() {
    if (!this.gaussian) return null;
    return this.memo(['sv2'], () => {
      const n = this.n, d = this.dim, h = n / 2;
      const hist = new Map();
      const range = [...Array(n).keys()].map((i) => i - h);
      const sq = range.map((v) => v * v);
      if (d === 2) {
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const m2 = sq[i] + sq[j]; if (m2) hist.set(m2, (hist.get(m2) || 0) + 1); }
      } else {
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const a = sq[i] + sq[j]; for (let l = 0; l < n; l++) { const m2 = a + sq[l]; if (m2) hist.set(m2, (hist.get(m2) || 0) + 1); } }
      }
      const m2s = [...hist.keys()];
      const ks = Float64Array.from(m2s, (m2) => this.kf * Math.sqrt(m2));
      const p = this.sim.linear_pk(ks, 1.0);
      let s = 0;
      m2s.forEach((m2, i) => { s += hist.get(m2) * p[i] / (ks[i] * ks[i]); });
      return s / d;      // box length L = 1
    });
  }
}

export { transpose2D };
