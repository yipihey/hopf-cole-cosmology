// engine.js - owns the CosmoSim (WASM) instance and every cached derived quantity.
//
// All getters take the *applied* parameter snapshot P (see app.js) and are
// memoised on (IC generation, relevant parameters). No DOM access here.

import { unpackSpectrum, unpackLoop, transpose2D } from '../hcc.js';
import { GpuCosmo3D } from '../gpu/cosmo3d.js';
import { GpuLpt3D } from '../gpu/lpt3d.js';
import { GpuSheet3D } from '../gpu/sheet3d.js';
import { getUnmergedSpec } from '../gpu/terms.js';
import { gpu3dRefusal } from './perf.js';
import { ENV } from './state.js';

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
    this.builtOrder = 0;           // highest order whose growth/term information is available (GPU path: growth tables, WASM path: LPT)
    this.wOrder = 0;               // order of the LPT built in the WASM sim (the GPU path builds none unless it falls back)
    this.cache = new Map();
    this.cacheBytes = 0;
    this.timings = new Map();
    this.dim = 2; this.n = 0;
    this.gaussian = false;
    this.kf = 2 * Math.PI; this.knyq = 0;
    // WebGPU compute path (3D only); see prepareGpu()
    this.gpuDev = null;           // GPUDevice shared with the viz layer (set by the app)
    this.g = null;                // GpuCosmo3D for the current grid
    this.lpt = null;              // GpuLpt3D: nLPT build, D_sc, Legendre split (all on the GPU)
    this.sheetG = null;           // GpuSheet3D: tetrahedral sheet density
    this.gTerms = null;           // {order, count} of the GPU-built raw LPT terms
    this.hsim = null;             // 8^3 helper CosmoSim: term list and growth functions g_tau(D) (grid independent), see ensureGrowth()
    this.hsOrder = 0; this.gSpec = null; this.gvCache = new Map();
    this.gFresh = {};             // field name -> memo key of the field currently resident in the GPU scratch buffers
    this.gPhiKey = null;          // which Legendre source the GPU 'phi-eff' buffer holds: 'lpt<order>|<D>' (see gpuLegendre)
    this.ref = null;              // GPU-side reference spectra of delta0 (cross spectra, linear theory)
    this.gpuError = null; this.gpuFailed = false; this.gpuRefused = false; this.gpuWanted = undefined;
    this.om = 1;                   // cosmology currently set on the sim (Omega_m)
    this.gq = Promise.resolve();   // serialises GPU work: the scratch buffers hold one field at a time
    this.lastPath = 'WASM';
  }

  // -- bookkeeping -------------------------------------------------------------

  time(label, fn) {
    const t0 = performance.now();
    try { return fn(); } finally { this.timings.set(label, performance.now() - t0); }
  }
  async timeAsync(label, fn) {
    const t0 = performance.now();
    try { return await fn(); } finally { this.timings.set(label, (this.timings.get(label) || 0) + performance.now() - t0); }
  }

  dispose() {
    this.releaseGpu();
    this.freeGrowthSim();
    if (this.sim) { try { this.sim.free(); } catch (e) { /* already freed */ } }
    this.sim = null; this.builtOrder = 0; this.wOrder = 0;
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
  hcCached(P) { return this.peek(this.hcKey(P)); }

  // -- Hopf-Cole source potentials ---------------------------------------------------------------
  //  'zel'  : Zel'dovich / Burgers (1LPT potential, hopf_cole)
  //  'lpt'  : Legendre transform of the order-P.order longitudinal displacement (hopf_cole_lpt, transverse = false)
  //  'lptT' : same with the first-order correction for the transverse part (orders >= 3)
  /** Source selector for a field name: 'hc' follows P.hs, 'hcz' is always Zel'dovich, 'hcl' is the Legendre variant. */
  hcSrcOf(which, P) {
    if (which === 'hcz') return 'zel';
    if (which === 'hcl') return P.hs === 'lptT' ? 'lptT' : 'lpt';
    return P.hs || 'zel';
  }
  static isHc(which) { return which === 'hc' || which === 'hcz' || which === 'hcl'; }
  /** Cache tag of a source: 'zel' | 'lpt<order>' | 'lptT<order>' (the T variant equals 'lpt' up to 2LPT, where Psi is a gradient). */
  hcTag(P, src = P.hs || 'zel') {
    if (src === 'zel') return 'zel';
    const o = Math.max(1, P.order);
    return (src === 'lptT' && o > 2 ? 'lptT' : 'lpt') + o;
  }
  hcKey(P, src) { return ['hc', P.D, P.nu, P.me, P.mx, this.hcTag(P, src)]; }

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
    this.gpuFailed = false; this.gpuError = null;
    this.om = 1;                    // a fresh CosmoSim is Einstein-de Sitter
    this.dim = dim; this.n = n;
    this.gaussian = P.ic === 'g';
    this.kf = this.sim.kf(); this.knyq = this.sim.knyq();
    this.builtOrder = 0; this.wOrder = 0;
    return true;
  }

  /** Switch the background cosmology (flat LCDM). Invalidates the LPT and everything derived from it; the IC is kept. */
  applyCosmology(om) {
    if (!this.sim || Math.abs(om - this.om) < 1e-12) return false;
    this.sim.set_cosmology(om);
    this.om = om; this.builtOrder = 0; this.wOrder = 0;
    // the raw LPT term fields S^tau on the GPU do not depend on the cosmology (only g_tau(D) does): they are kept,
    // the growth helper is rebuilt for the new cosmology, and everything derived from g_tau is dropped
    this.freeGrowthSim();
    this.gFresh = {}; this.gPhiKey = null;
    this.cache.clear(); this.cacheBytes = 0; this.gen++;    // also cancels in-flight GPU work (guarded by gen)
    return true;
  }
  /** The CosmoSim answering growth queries: the 8^3 helper on the 3-D GPU path (no WASM LPT is built there), else the sim itself. */
  gsim() { return this.useGpuLpt() ? this.hsim : this.sim; }
  /** Largest reachable D (Infinity in EdS; needs the growth tables). */
  dMax() { const s = this.gsim(); return s && this.builtOrder > 0 ? s.d_max() : Infinity; }
  /** Scale factor and Omega_m(a) at growth factor D. */
  cosmoAt(D) {
    const s = this.gsim();
    if (!s || this.builtOrder === 0) return { a: D, om: 1 };
    const a = s.a_of_d(D), o = this.om;
    const e2 = o / (a * a * a) + (1 - o);
    return { a, om: o / (a * a * a) / e2 };
  }
  /** LPT growth bookkeeping: [{label, order, g}] with g = g_tau(D)/D^order for the terms up to `order`. */
  termGrowth(D, order) {
    const s = this.gsim();
    if (!s || this.builtOrder === 0) return [];
    const labels = this.memo(['tlabels', this.builtOrder], () => Array.from(s.term_labels()));
    const orders = this.memo(['torders', this.builtOrder], () => Array.from(s.term_orders()));
    const g = s.term_growth(Math.max(D, 1e-6));
    const out = [], seen = {}, total = {};
    for (let t = 0; t < g.length; t++) total[orders[t]] = (total[orders[t]] || 0) + 1;
    for (let t = 0; t < g.length; t++) {
      if (orders[t] > order) continue;
      const k = seen[orders[t]] = (seen[orders[t]] || 0) + 1;
      out.push({ label: orders[t] + (total[orders[t]] > 1 ? String.fromCharCode(96 + k) : ''), full: labels[t], order: orders[t], g: g[t] });
    }
    return out;
  }
  /** Velocity-jump estimate max|u| - min|u| of the 1LPT (Zel'dovich) velocity; D-independent (u = S1). */
  velocityJump() {
    if (!this.sim || this.builtOrder < 1) return NaN;
    if (this.useGpuLpt()) { const v = this.peek(['du']); return v === undefined ? NaN : v; }   // seeded by prepareGpu
    return this.memo(['du'], () => {
      this.ensureWasm(1);
      const v = this.sim.velocities(1.0, 1), d = this.dim, N = v.length / d;
      let lo = Infinity, hi = 0;
      for (let i = 0; i < N; i++) {
        let s = 0; for (let a = 0; a < d; a++) { const x = v[i * d + a]; s += x * x; }
        s = Math.sqrt(s); if (s < lo) lo = s; if (s > hi) hi = s;
      }
      return hi - lo;
    });
  }

  /**
   * Make the LPT information for `order` available.  GPU path: only the growth tables/term list (helper sim, milliseconds); the
   * fields are built on the GPU by prepareGpu.  WASM path: build_lpt.
   */
  ensureLpt(order) {
    if (this.useGpuLpt()) {
      this.ensureGrowth(order);
      const ch = this.builtOrder < order;
      this.builtOrder = Math.max(this.builtOrder, order);
      return ch;
    }
    const ch = this.wOrder < order;
    this.ensureWasm(order);
    this.builtOrder = this.wOrder;
    return ch;
  }
  /** Build the LPT in the WASM sim (every WASM-only consumer calls this first; a no-op when already built). */
  ensureWasm(order) {
    if (this.wOrder >= order) return false;
    this.time('LPT' + order, () => this.sim.build_lpt(order));
    this.wOrder = order;
    if (!this.useGpuLpt()) this.builtOrder = Math.max(this.builtOrder, order);
    return true;
  }

  // -- growth helper (3D GPU path) -------------------------------------------------------------
  // The raw LPT term list and the growth functions g_tau(D) depend on (order, cosmology) only, not on the grid: a CosmoSim on
  // an 8^3 grid with build_lpt(order) provides term_specs_unmerged, EdS coefficients, term_g (EdS D^n or the LCDM tables), d_max,
  // a_of_d, labels - so the 3-D GPU path never has to build the (seconds-long) WASM LPT at the lab grid size.
  freeGrowthSim() {
    if (this.hsim) { try { this.hsim.free(); } catch (e) { /* ignore */ } }
    this.hsim = null; this.hsOrder = 0; this.gSpec = null; this.gvCache.clear();
  }
  ensureGrowth(order) {
    if (!this.hsim) {
      this.hsim = new this.core.CosmoSim(3, 8, 1.0);
      if (Math.abs(this.om - 1) > 1e-12) this.hsim.set_cosmology(this.om);
      this.hsOrder = 0;
    }
    if (this.hsOrder < order) {
      this.time('growth tables', () => { this.hsim.build_lpt(order); this.gSpec = getUnmergedSpec(this.hsim); });
      this.hsOrder = order; this.gvCache.clear();
    }
  }
  /** g_tau(D) of the GPU-built raw terms (EdS: c_tau D^order; LCDM: the integrated tables). */
  gGvals(D) {
    const key = String(D);
    let v = this.gvCache.get(key);
    if (v) return v;
    const spec = this.gSpec;
    v = spec.eds ? spec.orders.map((o, t) => spec.coefs[t] * Math.pow(D, o)) : Array.from(this.hsim.term_g(D));
    if (this.gvCache.size > 96) this.gvCache.clear();
    this.gvCache.set(key, v);
    return v;
  }
  /** True when the 3-D GPU path owns the LPT (no WASM build). */
  useGpuLpt() { return this.dim === 3 && !!this.gpuWanted && !!this.gpuDev && !this.gpuFailed && !!this.sim; }

  // -- WebGPU compute path (3D) ---------------------------------------------------------------
  //
  // The GPU path never changes the memo structure: it *seeds* the cache entries that the synchronous
  // getters below would otherwise compute with WASM (positions -> CIC, Hopf-Cole, P(k), r(k), Fourier
  // maps).  The consumers call `await eng.need(...)` first and then use the ordinary getters.

  /** Store a precomputed value under the memo key (no-op if present). */
  seed(parts, value) { return this.memo(parts, () => value); }

  /** Run `fn` after all previously queued GPU work (the GPU scratch buffers are shared by all fields). */
  gpuSerial(fn) {
    const p = this.gq.then(() => fn());
    this.gq = p.catch(() => {});
    return p;
  }

  /** True when the applied parameters P select the GPU path and it has not failed. */
  gpuActive(P) { return P.mode === 3 && !!P.gc && !!this.gpuDev && !this.gpuFailed && this.dim === 3; }

  releaseGpu() {
    for (const o of [this.sheetG, this.lpt]) if (o && o.destroy) { try { o.destroy(); } catch (e) { console.warn('[engine] GPU destroy:', e); } }
    if (this.g) { try { this.g.destroy(); } catch (e) { console.warn('[engine] GPU destroy:', e); } }
    const had = !!this.g || !!this.ref;
    this.g = null; this.lpt = null; this.sheetG = null; this.gTerms = null; this.ref = null; this.gFresh = {}; this.gPhiKey = null;
    // GPU-derived cache entries and in-flight GPU work (guarded by `gen`) are stale now
    if (had) { this.cache.clear(); this.cacheBytes = 0; this.gen++; }
  }

  /** Switch the GPU path on/off (clears the cache so that the other path recomputes and its timings show). */
  setGpuPath(on) {
    if (on === this.gpuWanted) return;
    this.gpuWanted = on;
    this.gpuFailed = false; this.gpuError = null;
    if (!on) this.releaseGpu();
    this.builtOrder = on ? 0 : this.wOrder;       // the owner of the LPT changes: ensureLpt re-establishes it
    this.cache.clear(); this.cacheBytes = 0; this.gen++;
  }

  /**
   * Prepare the GPU path for the applied parameters: create the GPU objects for the current grid (once), the FFT of delta0 for the
   * reference spectra (once per IC), and build the raw LPT term fields S^tau on the GPU from delta0 (once per IC and order
   * increase; changing the cosmology or lowering the order needs no rebuild).  Returns true when the GPU path is ready; on any
   * failure the GPU path is disabled for this IC and the WASM path takes over.
   */
  prepareGpu(P) { return this.gpuSerial(() => this._prepareGpu(P)); }
  async _prepareGpu(P) {
    if (!this.gpuActive(P) || !this.sim) { this.lastPath = 'WASM'; return false; }
    const gen = this.gen;
    try {
      // memory safety: refuse (-> WASM, with a message) before any GPU buffer is created
      const refusal = gpu3dRefusal(this.gpuDev, this.n, P.order, ENV.lite);
      if (refusal) throw Object.assign(new Error(refusal), { refusal: true });
      if (!this.g || this.g.n !== this.n) {
        this.releaseGpu();
        await this.timeAsync('GPU init', async () => {
          const g = new GpuCosmo3D(this.gpuDev, this.n); await g.init();
          const lpt = new GpuLpt3D(g); await lpt.init();
          const sh = new GpuSheet3D(g); await sh.init();
          this.g = g; this.lpt = lpt; this.sheetG = sh;
        });
      }
      const g = this.g;
      this.ensureLpt(P.order);
      if (!this.ref) {
        await this.timeAsync('GPU ref', async () => {
          const d0 = this.sim.linear_delta(1.0);
          const an = await g.setReference(d0, this.nbins);      // FFT kept on the GPU for cross spectra; maps
          // linear P(k) from the f64 CPU FFT (once per IC): the f32 GPU FFT is noisy where P < 1e-17 of its peak
          const pk = unpackSpectrum(this.sim.power_spectrum(d0, this.nbins, false));
          this.ref = { d0, pk, maps: an.maps };
        });
      }
      if (!this.gTerms || this.gTerms.order < P.order) {
        await this.timeAsync('GPU nLPT build', async () => {
          const r = await this.lpt.build(this.ref.d0, this.gSpec, P.order);
          this.gTerms = { order: P.order, count: r.count };
          this.gPhiKey = null;               // phi-eff of the previous build is stale
        });
      }
      if (gen === this.gen && this.peek(['du']) === undefined) {
        const du = await this.timeAsync('|S1| range', () => this.lpt.velocityJump());
        if (gen === this.gen) this.seed(['du'], du);
      }
      if (gen !== this.gen) return false;
      this.lastPath = 'GPU';
      return true;
    } catch (err) {
      if (gen !== this.gen) return false;      // the IC changed under us (buffers destroyed): not a GPU failure
      this.gpuFailure(err);
      return false;
    }
  }

  /** Immediately-needed density field on the GPU path (CIC, tetrahedral sheet or Hopf-Cole), read back to the CPU and cached. */
  async _gpuField(which, P) {
    if (!this.gpuActive(P) || !this.g || !this.gTerms) return;
    const g = this.g, gen = this.gen;
    const order = Math.min(P.order, this.gTerms.order);
    try {
      if (which === 'cic') {
        const key = ['cic', P.D, P.order];
        if (this.peek(key) !== undefined) return;
        const rho = await this.timeAsync('CIC (GPU)', async () => {
          const pos = g.positions(this.gGvals(P.D), order);
          return g.readField(g.cicDensity(pos));
        });
        if (gen !== this.gen) return;
        this.seed(key, rho);
        this.gFresh.cic = key.join('|');
      } else if (which === 'sheet') {
        const key = ['sheet', P.D, P.order];
        if (this.peek(key) !== undefined) return;
        const rho = await this.timeAsync('sheet (tetra, GPU)', async () => {
          const disp = g.displacement(this.gGvals(P.D), order);
          return g.readField(this.sheetG.density(disp));
        });
        if (gen !== this.gen) return;
        this.seed(key, rho);
        this.gFresh.sheet = key.join('|');
      } else if (Engine.isHc(which) && P.me === 1) {
        const src = this.hcSrcOf(which, P), tag = this.hcTag(P, src);
        const key = this.hcKey(P, src);
        if (this.peek(key) !== undefined) return;
        const trans = tag.startsWith('lptT');
        let range = 0;
        const delta = await this.timeAsync(tag === 'zel' ? 'Hopf-Cole (GPU)' : trans ? 'Hopf-Cole (GPU, Legendre + transverse)' : 'Hopf-Cole (GPU, Legendre)', async () => {
          const psrc = tag === 'zel' ? 'phi0' : 'phi-eff';
          if (tag !== 'zel') await this.gpuLegendre(P, order, gen);
          if (gen !== this.gen) return null;
          const enc = this.gpuDev.createCommandEncoder({ label: 'hopf-cole' });
          const hc = g.hopfCole(P.nu, P.D, enc, psrc);
          if (trans) this.lpt.transverse(P.D, enc);
          this.gpuDev.queue.submit([enc.finish()]);
          const d = await g.readField(hc.delta);
          range = g.hcExponentRange(P.nu, psrc);
          return d;
        });
        if (gen !== this.gen || !delta) return;
        this.seed(key, { delta, nuEff: P.nu, range, floor: 1 / (this.n * this.n * 4 * Math.max(P.D, 1e-12)), ratio: NaN });
        this.gFresh.hc = key.join('|');              // the single hc scratch buffer now holds this field
      }
    } catch (err) { if (gen === this.gen) this.gpuFailure(err); }
  }

  /**
   * Make the GPU 'phi-eff' buffer hold the effective longitudinal potential of the order-`order` map at P.D (Psi_L = -D grad phi_eff;
   * Helmholtz split of Psi(D) = sum g_tau S^tau on the GPU, also giving Psi_T) and cache rms|Psi_T|/rms|Psi_L|.
   * It depends on D, order and cosmology (the latter resets gPhiKey), not on nu.
   */
  async gpuLegendre(P, order, gen) {
    const want = 'lpt' + order + '|' + P.D;
    if (this.gPhiKey === want && this.g.pstat['phi-eff']) return;
    const r = await this.timeAsync('Legendre split (GPU)', () => this.lpt.legendre(this.gGvals(P.D), order, P.D));
    if (gen !== this.gen) return;
    this.gPhiKey = want;
    this.seed(['psit', P.D, P.order], r.frac);
  }

  /** First shell-crossing times D_sc(P.order) and D_sc(1) on the GPU (batched Jacobian minima), cached in the memo. */
  async needDsc(P) {
    if (!this.gpuActive(P) || !this.g || !this.gTerms) return;
    await this.gpuSerial(() => this._gpuDsc(P));
  }
  async _gpuDsc(P) {
    const gen = this.gen, dmax = this.dMax();
    try {
      for (const o of new Set([P.order, 1])) {
        if (this.peek(['dsc', o]) !== undefined) continue;
        const v = await this.timeAsync('D_sc' + o + ' (GPU)', () => this.lpt.shellCrossing(Math.min(o, this.gTerms.order), (D) => this.gGvals(D), dmax));
        if (gen !== this.gen) return;
        this.seed(['dsc', o], v);
      }
    } catch (err) { if (gen === this.gen) this.gpuFailure(err); }
  }

  /**
   * One GPU FFT of the named field and everything derived from it: P(k) (plain and CIC-deconvolved),
   * r(k) against the linear field and the Fourier maps, all seeded into the cache.
   */
  async _gpuAnalysis(which, P) {
    if (!this.gpuActive(P) || !this.g || !this.ref || which === 'lin') return;
    const gen = this.gen;
    const fk = this.fieldKey(which, P);
    const keyA = ['gpuan', ...fk];
    if (this.peek(keyA) !== undefined) return;
    try {
      await this._gpuField(which, P);
      if (gen !== this.gen || !this.g) return;
      // use the GPU-resident field when it is the one just computed (no CPU round trip), else upload the cached CPU copy
      const g = this.g, fkey = this.fieldKey(which, P).join('|');
      let buf, offset = 0;
      const hcLike = Engine.isHc(which);
      if (which === 'cic' && this.gFresh.cic === fkey) { buf = g.buf('rho'); offset = 1; }
      else if (which === 'sheet' && this.gFresh.sheet === fkey) { buf = g.buf('sheet-rho'); offset = 1; }
      else if (hcLike && this.gFresh.hc === fkey) buf = g.buf('hc-delta');
      else buf = g.uploadField(this.delta(which, P));
      const an = await this.timeAsync('FFT + spectra (GPU)', async () => g.analyze(buf, { nbins: this.nbins, maps: true, cross: true, offset }));
      if (gen !== this.gen) return;
      const D = P.D, p0 = this.ref.pk.p;
      const r = new Float64Array(an.cross.k.length);
      for (let i = 0; i < r.length; i++) r[i] = (D * an.cross.p[i]) / Math.sqrt(Math.max(1e-300, an.plain.p[i] * D * D * p0[i]));
      this.seed(['pk', ...fk, false], an.plain);
      if (which === 'cic') this.seed(['pk', ...fk, true], an.dec);
      this.seed(['rk', ...fk], { k: an.cross.k, r });
      this.seed(['fmap', ...fk], an.maps);
      this.seed(keyA, 1);
    } catch (err) { if (gen === this.gen) this.gpuFailure(err); }
  }

  gpuFailure(err) {
    (err && err.refusal ? console.warn : console.error)('[engine] GPU compute path failed, falling back to WASM:', err);
    this.gpuError = (err && err.message) || String(err);
    this.gpuRefused = !!(err && err.refusal);       // a planned memory-safety fallback, not a malfunction
    this.gpuFailed = true;
    this.lastPath = 'WASM';
    this.releaseGpu();
    this.builtOrder = this.wOrder;          // the WASM sim owns the LPT again (consumers build it on demand)
  }

  /**
   * Consumer entry point: `await eng.need('field' | 'analysis', which, P)` makes the (expensive) result for
   * `which` available in the cache.  A no-op on the WASM path, where the synchronous getters compute lazily.
   */
  async need(what, which, P) {
    if (!this.gpuActive(P) || !this.g) return;
    if (what === 'field') await this.gpuSerial(() => this._gpuField(which, P));
    else if (what === 'analysis') await this.gpuSerial(() => this._gpuAnalysis(which, P));
  }

  /** Label of the path that served the last run, for the status line. */
  pathLabel(P) {
    if (P.mode !== 3) return '';
    if (this.gpuError) return 'WASM (GPU failed: ' + this.gpuError + ')';
    return this.lastPath === 'GPU' ? 'GPU compute' : 'WASM';
  }

  // -- fields --------------------------------------------------------------------

  positions(P) { this.ensureWasm(P.order); return this.sim.positions(P.D, Math.min(P.order, this.wOrder)); }

  sheet(P) {
    // 2D: 2x2 supersampled triangles; 3D: six Kuhn tetrahedra per Lagrangian cell, point-sampled (GPU path: seeded by _gpuField; WASM ~1 s at 64^3)
    return this.memo(['sheet', P.D, P.order], () => { this.ensureWasm(P.order); return this.time(this.dim === 3 ? 'sheet (tetra, CPU)' : 'sheet', () =>
      this.sim.sheet_density(P.D, P.order, this.n, this.dim === 3 ? 1 : 2)); });
  }
  cic(P) {
    return this.memo(['cic', P.D, P.order], () => { this.ensureWasm(P.order); return this.time('CIC', () => this.sim.cic_density(P.D, P.order, this.n)); });
  }
  linear(P) {
    // GPU path: delta_lin = D delta0 exactly, so reuse the cached delta0 instead of an FFT per D
    if (this.ref) return this.memo(['lin', P.D], () => { const d0 = this.ref.d0, o = new Float32Array(d0.length); for (let i = 0; i < o.length; i++) o[i] = P.D * d0[i]; return o; });
    return this.memo(['lin', P.D], () => this.time('linear', () => this.sim.linear_delta(P.D)));
  }
  /** Hopf-Cole solution for the source potential `src` (default: the selected one, P.hs). */
  hc(P, src = P.hs || 'zel') {
    const tag = this.hcTag(P, src);
    return this.memo(this.hcKey(P, src), () => this.time(tag === 'zel' ? 'Hopf-Cole' : 'Hopf-Cole (Legendre)', () => {
      const s = this.sim;
      let ratio = 0;
      if (tag === 'zel') s.hopf_cole(P.D, P.nu, P.me, P.mx);
      else { this.ensureWasm(P.order); ratio = s.hopf_cole_lpt(P.D, P.order, P.nu, P.me, P.mx, tag.startsWith('lptT')); }
      const r = { delta: s.hc_delta(), nuEff: s.hc_nu_eff(), range: s.hc_exponent_range(), floor: s.hc_nu_floor(), ratio };
      if (this.dim === 2) {
        r.phi = s.hc_phi(); r.lnpsi = s.hc_lnpsi(); r.psihat = s.hc_psihat_log(); r.vel = s.hc_velocity();
      }
      return r;
    }));
  }
  /** rms|Psi_T| / rms|Psi_L| of the order-P.order displacement at D (0 up to 2LPT); D-dependent, ν-independent. */
  psiT(P) {
    if (P.order <= 2) return 0;
    return this.memo(['psit', P.D, P.order], () => {       // GPU path: seeded by gpuLegendre
      this.ensureWasm(P.order);
      return this.time('Psi_T/Psi_L', () => this.sim.hopf_cole_lpt(P.D, P.order, 1e-2, 0, 30, true));
    });
  }
  jacobian(P) {
    return this.memo(['jac', P.D, P.order], () => { this.ensureWasm(P.order); return this.time('Jacobian', () => this.sim.jacobian(P.D, P.order)); });
  }
  div(P, nth) {
    this.ensureWasm(Math.max(nth, P.order));
    return this.memo(['div', nth], () => this.sim.lpt_div(nth));
  }
  curl(P, nth) {
    this.ensureWasm(Math.max(nth, P.order));
    return this.memo(['curl', nth], () => this.sim.lpt_curl(nth, 0));
  }
  /** First shell-crossing D for the given order (Infinity if none). */
  dsc(order) {
    return this.memo(['dsc', order], () => { this.ensureWasm(order); return this.time('D_sc' + order, () => this.sim.shell_crossing(order)); });   // GPU path: seeded by needDsc
  }

  /** Parameter parts identifying a density-like field. */
  fieldKey(which, P) {
    switch (which) {
      case 'sheet': case 'cic': return [which, P.D, P.order];
      case 'hc': case 'hcz': case 'hcl': return this.hcKey(P, this.hcSrcOf(which, P));
      default: return ['lin', P.D];
    }
  }
  /** Overdensity delta (mean 0) of the named field. */
  delta(which, P) {
    return this.memo(['delta', ...this.fieldKey(which, P)], () => {
      if (Engine.isHc(which)) return this.hc(P, this.hcSrcOf(which, P)).delta;
      if (which === 'lin') return this.linear(P);
      const rho = which === 'sheet' ? this.sheet(P) : this.cic(P);
      const o = new Float32Array(rho.length);
      for (let i = 0; i < o.length; i++) o[i] = rho[i] - 1;
      return o;
    });
  }
  /** Density 1+delta (3D visualisation); linear theory clipped at 1e-3 so it can be shown on a log scale. */
  rho(which, P) {
    return this.memo(['rho', ...this.fieldKey(which, P)], () => {
      if (which === 'cic') return this.cic(P);
      const d = this.delta(which, P), o = new Float32Array(d.length);
      for (let i = 0; i < o.length; i++) { const v = d[i] + 1; o[i] = v > 1e-3 ? v : 1e-3; }
      return o;
    });
  }
  /**
   * Top-hat smoothed overdensity (sphere in 3D, disc in 2D) of diameter `diam` cells, via the Fourier-space window (WASM
   * CosmoSim.tophat_smooth).  The 3D GPU path seeds this entry with the GPU FFT (needSmooth) beforehand.
   */
  smooth(which, P, diam) {
    return this.memo(['smooth', ...this.fieldKey(which, P), diam], () => this.time('top-hat smoothing', () => this.sim.tophat_smooth(this.delta(which, P), diam / 2)));
  }
  /** GPU path: smooth the (cached) overdensity of `which` on the GPU and seed the memo entry read by smooth(). No-op elsewhere. */
  async needSmooth(which, P, diam) {
    if (!this.gpuActive(P) || !this.g) return;
    await this.gpuSerial(() => this._gpuSmooth(which, P, diam));
  }
  async _gpuSmooth(which, P, diam) {
    if (!this.gpuActive(P) || !this.g) return;
    const key = ['smooth', ...this.fieldKey(which, P), diam];
    if (this.peek(key) !== undefined) return;
    const gen = this.gen, g = this.g;
    try {
      const d = this.delta(which, P);
      const out = await this.timeAsync('top-hat (GPU)', () => g.smoothTophat(g.uploadField(d), diam / 2));
      if (gen !== this.gen) return;
      this.seed(key, out);
    } catch (err) { if (gen === this.gen) this.gpuFailure(err); }
  }
  /** Fourier maps: amp = log10|f^|/max, phase; both fft-shifted, as [ikx*n+iky]. */
  fmaps(which, P) {
    return this.memo(['fmap', ...this.fieldKey(which, P)], () => this.time('Fourier maps', () => {
      if (which === 'lin' && this.ref && P.D > 0) return this.ref.maps;     // |f^|/max and arg are scale invariant
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
    return this.memo(['pk', ...this.fieldKey(which, P), dc], () => this.time('P(k)', () => {
      if (which === 'lin' && this.ref) {       // D^2 P(k) of delta0, from the GPU reference analysis
        const D2 = P.D * P.D, r = this.ref.pk;
        return { k: r.k, p: Float64Array.from(r.p, (v) => v * D2), n: r.n };
      }
      return unpackSpectrum(this.sim.power_spectrum(this.delta(which, P), this.nbins, dc));
    }));
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
    // P_lin ~ D^2, P22 and P13 ~ D^4 exactly: evaluate once at D = 1 and rescale
    const base = this.memo(['loop1', kernels], () => this.time(kernels ? '1-loop ZA' : '1-loop SPT', () => {
      const ks = this.pk('lin', { ...P, D: 1 }, false).k;
      let m = 0; while (m < ks.length && ks[m] <= this.knyq) m++;
      const k = Float64Array.from(ks.subarray(0, m));
      return { k, ...unpackLoop(this.sim.one_loop(k, 1.0, kernels)) };
    }));
    const D = P.D, d2 = D * D, d4 = d2 * d2;
    return this.memo(['loop', D, kernels], () => ({
      k: base.k,
      plin: Float64Array.from(base.plin, (v) => v * d2),
      p22: Float64Array.from(base.p22, (v) => v * d4),
      p13: Float64Array.from(base.p13, (v) => v * d4),
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
