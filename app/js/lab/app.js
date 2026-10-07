// app.js - orchestrates state, engine, controls, field panels and spectra.
//
// Computation model: every user-visible computation is a *task* in a chain
// (see run()). Between tasks control returns to the browser (so the status
// line and the already finished panels repaint), and a newer run() cancels the
// older chain at the next task boundary. Results are memoised in the Engine,
// so toggling panels or series never recomputes anything that is cached.

import { getGPU } from '../viz/gpu.js';
import { Engine } from './engine.js';
import { decodeHash, encodeHash, defaultN, defaultLive, ENV } from './state.js';
import { buildControls } from './controls.js';
import { Fields2D } from './fields2d.js';
import { Fields3D } from './fields3d.js';
import { Spectra } from './spectra.js';
import { LegendreLab } from './legendre.js';
import { buildExplain } from './explain.js';
import { el, tick, paint, fmtMs } from './dom.js';

const $ = (id) => document.getElementById(id);

export class Lab {
  constructor(core) {
    this.core = core;
    this.S = decodeHash(location.hash);
    this.P = structuredClone(this.S);
    this.eng = new Engine(core);
    this.runGen = 0;
    this.timers = {};
    this.dirty = false;
    this.fields = null; this.fieldsMode = 0; this.fieldsN = 0;
    this.spectra = null; this.legendre = null;
    window.addEventListener('hashchange', () => { if (location.hash.replace(/^#/, '') !== encodeHash(this.S)) location.reload(); });
  }

  async start() {
    const gpu = await getGPU();
    this.gpu = !!gpu;
    this.backend = this.gpu ? 'webgpu' : 'canvas2d';
    // WebGPU compute path (3D): needs the shared device; default 'live update' depends on it
    this.gpuCompute = !!gpu;
    ENV.gpuCompute = this.gpuCompute;
    this.eng.gpuDev = gpu ? gpu.device : null;
    if (!new URLSearchParams(location.hash.replace(/^#/, '')).has('lv')) { this.S.live = defaultLive(this.S.mode, this.S.gc); this.P.live = this.S.live; }
    if (new URLSearchParams(location.search).has('embed')) document.body.classList.add('embed');
    this.statusEl = $('lab-status-text');
    this.timingEl = $('lab-status-timing');
    $('lab-status-backend').textContent = `backend: ${this.backend}`;

    this.controls = buildControls(this, $('lab-controls'));
    this.buildVisChips();
    this.spectra = new Spectra(this, $('spectra-host'));
    this.legendre = new LegendreLab(this, $('legendre-host'));
    buildExplain($('explain-host'));
    this.applyVisibility();
    this.ensureFields(this.S);
    window.__lab = this;       // debugging / tests
    await this.run();
  }

  // ---------------------------------------------------------------- state plumbing

  hashChanged() {
    clearTimeout(this.timers.hash);
    this.timers.hash = setTimeout(() => {
      const h = encodeHash(this.S);
      try { history.replaceState(null, '', location.pathname + location.search + (h ? '#' + h : '')); } catch (e) { /* sandboxed */ }
    }, 350);
  }

  setStatus(text, kind = '') {
    this.statusEl.textContent = text;
    this.statusEl.className = kind;
    $('lab-status-dot').className = 'lab-dot ' + kind;
  }

  reportError(label, err) {
    console.error(`[lab] ${label}:`, err);
    this.lastError = `error in ${label}: ${err && err.message ? err.message : err}`;
    this.setStatus(this.lastError, 'err');
  }

  /** Called by controls for every changed parameter. */
  setParam(key, value, level) {
    const S = this.S;
    S[key] = value;
    this.hashChanged();
    if (level === 'live') {
      if (value) this.schedule(60);
      else this.setDirty(false);
      return;
    }
    if (level === 'nu') {
      // fast nu path: only the Hopf-Cole dependent results are recomputed (everything else is cached)
      if (!S.live) { this.invalidate(); this.setDirty(true); this.markStaleHc(true); return; }
      this.markStaleHc(true);
      this.setStatus('ν changing …', 'busy');
      this.scheduleHc();
      return;
    }
    this.invalidate();
    this.appliedKey = null;
    if (key === 'D') this.P.D = this.S.D;   // fast path works on the applied snapshot
    if (!S.live) {
      this.setDirty(true);
      this.markStale(true);
      return;
    }
    if (level === 'dyn') {
      this.markStale(true, true);
      this.fastD();
      this.schedule(170);
    } else {
      this.markStale(true);
      this.schedule(level === 'ic' ? 70 : 110);
    }
  }

  /** The "GPU compute" checkbox (3D). Live update follows its default; the other path recomputes. */
  setGpuCompute(v) {
    const S = this.S;
    S.gc = v;
    S.live = defaultLive(S.mode, v);
    this.hashChanged();
    this.controls.syncAll();
    this.invalidate();
    this.markStale(true);
    if (S.mode === 3 && this.eng.sim) {
      if (S.live) this.schedule(30); else { this.setDirty(true); this.setStatus('GPU compute off: press Run (the WASM path takes seconds at 128³).'); }
    }
  }
  /** True when the GPU compute path is requested and possible for the current state. */
  gpuComputeOn(S = this.S) { return this.gpuCompute && S.mode === 3 && !!S.gc; }

  setMode(m) {
    if (m === this.S.mode) return;
    const S = this.S;
    this.eng.releaseGpu();
    S.mode = m; S.n = defaultN(m); S.live = defaultLive(m, S.gc);
    this.appliedKey = null;
    // 3D: 'sheet' is the tetrahedral sheet; the measured CIC spectrum is the default there
    if (m === 3 && S.ser.includes('sheet') && !S.ser.includes('cic')) S.ser = S.ser.map((q) => (q === 'sheet' ? 'cic' : q));
    else if (m === 2 && S.ser.join() === 'lin,cic,hc,spt') S.ser = ['lin', 'sheet', 'hc', 'spt'];
    if (m === 3 && S.me > 1) S.me = 1;
    this.invalidate();
    this.controls.syncAll();
    this.spectra.syncMode();
    this.legendre.syncMode();
    this.hashChanged();
    this.markStale(true);
    this.ensureFields(S);
    this.eng.timings.clear(); this.showTimings();
    if (m === 3) {
      this.setDirty(true);
      this.setStatus('3D: choose parameters, then press Run.');
    } else this.schedule(30);
  }

  setDirty(b) { this.dirty = b; this.controls.setDirty(b); }

  invalidate() { this.runGen++; document.body.classList.remove('lab-working'); }

  schedule(ms) {
    clearTimeout(this.timers.run);
    this.timers.run = setTimeout(() => this.run(), ms);
  }

  runNow() { clearTimeout(this.timers.run); this.run(); }

  markStale(b, keepFast = false) {
    if (this.fields) this.fields.markStale(b, keepFast);
    if (this.spectra) this.spectra.markStale(b);
    if (this.legendre) this.legendre.markStale(b);
  }

  markStaleHc(b) {
    if (this.fields && this.fields.markStaleHc) this.fields.markStaleHc(b);
    if (this.spectra) this.spectra.markStale(b);
    if (this.legendre) this.legendre.markStale(b);
  }

  /** Key of everything the non-Hopf-Cole results depend on (a nu change leaves it unchanged). */
  coreKey(S) { return [Engine.icSignature(S), S.D, S.order, S.om, S.gc, S.mode, S.n].join('|'); }

  scheduleHc() {
    clearTimeout(this.timers.hc);
    this.timers.hc = setTimeout(() => this.runHc(), this.S.mode === 3 ? 40 : 80);
  }

  /**
   * Fast nu path: recompute only the Hopf-Cole dependent panels and spectra. The memo cache is keyed
   * (IC, D, order) for LPT / sheet / CIC results and (IC, D, nu, method) for Hopf-Cole results, so nothing
   * else is touched. Falls back to a full run when something else changed since the last full run.
   */
  async runHc() {
    clearTimeout(this.timers.hc);
    const S = this.S, e = this.eng;
    if (!e.sim || this.appliedKey === null || this.appliedKey !== this.coreKey(S)) { this.run(); return; }
    const gen = ++this.runGen;
    this.P = structuredClone(S);
    const P = this.P;
    this.setDirty(false);
    this.lastError = null;
    const t0 = performance.now();
    e.timings.clear();
    const tasks = [];
    if (this.visible('f')) tasks.push(...this.fields.tasks(P, true).map((t) => ({ ...t, heavy: false })));
    if (this.visible('s')) tasks.push(...this.spectra.tasks(P).map((t) => ({ ...t, heavy: false })));
    if (this.visible('l')) tasks.push(...this.legendre.tasks(P).map((t) => ({ ...t, heavy: false })));
    tasks.push({ label: 'readouts', fn: () => this.refreshReadouts() });
    const ok = await this.runTasks(tasks, gen);
    if (gen !== this.runGen) return;
    this.refreshReadouts();
    if (this.lastError) this.setStatus(this.lastError, 'err');
    else if (ok) this.setStatus(`ready · ν = ${P.nu.toExponential(1)} · Hopf–Cole only · ${P.mode}D ${P.n}${P.mode === 2 ? '²' : '³'} · D = ${P.D.toFixed(3)}${e.pathLabel(P) ? ' · ' + e.pathLabel(P) : ''} · ${fmtMs(performance.now() - t0)}`);
    this.showTimings();
  }

  // ---------------------------------------------------------------- sections

  buildVisChips() {
    const host = $('lab-vis');
    const defs = [['f', 'Fields'], ['s', 'Spectra'], ['l', 'Legendre lab'], ['e', 'Explain']];
    this.visCbs = {};
    el('span', 'hcc-label', host, 'Show:');
    for (const [k, label] of defs) {
      const lab = el('label', 'hcc-ctl hcc-check lab-chip', host);
      const cb = el('input', null, lab); cb.type = 'checkbox'; cb.checked = this.S.vis.includes(k);
      el('span', 'hcc-label', lab, label);
      this.visCbs[k] = cb;
      cb.addEventListener('change', () => {
        this.S.vis = defs.map(([q]) => q).filter((q) => this.visCbs[q].checked).join('');
        this.hashChanged();
        this.applyVisibility();
        if (cb.checked && this.eng.sim && (k === 'f' || k === 's' || k === 'l')) this.runSection(k);
      });
    }
  }
  applyVisibility() {
    const v = this.S.vis;
    $('sec-fields').hidden = !v.includes('f');
    $('sec-spectra').hidden = !v.includes('s');
    $('sec-legendre').hidden = !v.includes('l');
    $('sec-explain').hidden = !v.includes('e');
  }
  visible(k) { return this.S.vis.includes(k); }

  ensureFields(P) {
    const n = P.n, mode = P.mode;
    if (this.fields && this.fieldsMode === mode && (mode === 2 || this.fieldsN === n)) return;
    if (this.fields) this.fields.destroy();
    const host = $('fields-host');
    this.fields = mode === 2 ? new Fields2D(this, host) : new Fields3D(this, host);
    this.fieldsMode = mode; this.fieldsN = n;
  }

  // ---------------------------------------------------------------- running

  async runTasks(tasks, gen) {
    for (const t of tasks) {
      if (gen !== this.runGen) return false;
      this.setStatus(`computing: ${t.label} …`, 'busy');
      document.body.classList.add('lab-working');
      await (t.heavy ? paint() : tick());
      if (gen !== this.runGen) return false;
      try {
        await t.fn();
      } catch (err) {
        this.reportError(t.label, err);
        if (t.critical) { document.body.classList.remove('lab-working'); return false; }
      }
    }
    document.body.classList.remove('lab-working');
    return true;
  }

  /** Full (re)computation for the current control state. */
  async run() {
    clearTimeout(this.timers.run);
    const gen = ++this.runGen;
    const S = this.S;
    this.P = structuredClone(S);
    const P = this.P;
    this.setDirty(false);
    this.lastError = null;
    const t0 = performance.now();
    this.eng.timings.clear();
    this.ensureFields(P);
    const icChanged = !this.eng.sim || Engine.icSignature(P) !== this.eng.sig;
    const heavy3 = P.mode === 3;

    const tasks = [];
    clearTimeout(this.timers.hc);
    this.appliedKey = null;
    this.eng.setGpuPath(this.gpuComputeOn(P));
    tasks.push({ label: 'initial conditions', heavy: icChanged, critical: true, fn: () => { this.eng.configure(P); this.eng.applyCosmology(P.om); } });
    const gpuOn = this.gpuComputeOn(P);
    // GPU path: the LPT task only sets up the growth tables (milliseconds); the fields are built on the GPU below
    tasks.push({ label: gpuOn ? `growth tables, order ${P.order}` : `LPT order ${P.order}`, heavy: !gpuOn && (icChanged || this.eng.builtOrder < P.order || P.om !== this.eng.om), critical: true,
      fn: () => { this.eng.ensureLpt(P.order); this.limitD(P); } });
    if (heavy3) tasks.push({ label: 'GPU nLPT build', heavy: false, fn: () => this.eng.prepareGpu(P) });
    if (this.visible('f')) {
      const ft = this.fields.tasks(P);
      tasks.push(...ft.filter((t) => !t.heavy), ...ft.filter((t) => t.heavy));
    }
    if (this.visible('s')) tasks.push(...this.spectra.tasks(P));
    if (this.visible('l')) tasks.push(...this.legendre.tasks(P));
    tasks.push({ label: 'shell-crossing times', heavy: heavy3 && !gpuOn && this.eng.peek(['dsc', P.order]) === undefined,
      fn: async () => { await this.eng.needDsc(P); this.eng.dsc(P.order); this.eng.dsc(1); this.refreshReadouts(); } });

    const ok = await this.runTasks(tasks, gen);
    if (gen !== this.runGen) return;
    this.refreshReadouts();
    if (ok && !this.lastError) this.appliedKey = this.coreKey(P);
    if (this.lastError) this.setStatus(this.lastError, 'err');
    else if (ok) {
      const path = this.eng.pathLabel(P);
      this.setStatus(`ready · ${P.mode}D ${P.n}${P.mode === 2 ? '²' : '³'} · order ${P.order} · D = ${P.D.toFixed(3)}${path ? ' · ' + path : ''} · ${fmtMs(performance.now() - t0)}`, this.eng.gpuError && P.mode === 3 ? 'err' : '');
    }
    this.showTimings();
  }

  /** Compute only the tasks of one section (when it is switched on). */
  async runSection(k) {
    const gen = this.runGen;
    const P = this.P;
    const tasks = k === 'f' ? this.fields.tasks(P) : k === 'l' ? this.legendre.tasks(P) : this.spectra.tasks(P);
    await this.runTasks(tasks, gen);
    if (gen === this.runGen) { if (this.lastError) this.setStatus(this.lastError, 'err'); else this.setStatus('ready'); this.showTimings(); }
  }

  runSpectra() { if (this.eng.sim && this.eng.dim === this.S.mode && this.visible('s')) this.runSection('s'); }

  /** Re-render one field panel from cache (display-only change). */
  async runPanel(owner, idx) {
    if (!this.eng.sim || this.eng.dim !== this.fieldsMode) return;
    const gen = this.runGen;
    this.setStatus('drawing panel …', 'busy');
    await tick();
    try {
      if (owner.update && owner.slots === undefined) await owner.update(idx, this.P);
      else await owner.slots[idx].update(this.P);
      if (gen === this.runGen) { this.setStatus('ready'); this.showTimings(); }
    } catch (err) { this.reportError('panel', err); }
  }

  /** D slider: reposition the sheet immediately (cheap), the rest follows after the debounce. */
  fastD() {
    this.P.D = this.S.D;
    if (this.fastQueued) return;
    this.fastQueued = true;
    requestAnimationFrame(() => {
      this.fastQueued = false;
      const e = this.eng, S = this.S;
      if (!e.sim || e.sig !== Engine.icSignature(S) || e.builtOrder < this.P.order) return;
      try {
        const t0 = performance.now();
        this.fields.fastUpdate(this.P);
        this.fastMs = performance.now() - t0;
        this.refreshReadouts();
      } catch (err) { this.reportError('fast sheet update', err); }
    });
    this.setStatus('D changing …', 'busy');
  }

  /** In flat LCDM the D slider ends at 0.999 D_max; clamp the applied and the displayed D. */
  limitD(P) {
    const eff = this.controls.setDMax(this.eng.dMax());
    if (P.D > eff) { P.D = eff; this.S.D = eff; this.controls.D.set(eff); this.hashChanged(); }
  }

  refreshReadouts() {
    const e = this.eng;
    if (!e.sim || !this.controls) return;
    this.controls.setCosmo(e.builtOrder > 0 ? e.cosmoAt(this.P.D) : null, e.dMax(), e.termGrowth(this.P.D, this.P.order));
    {
      const hc = e.hcCached(this.P), du = e.velocityJump();
      const nu = hc ? hc.nuEff : this.P.nu;
      this.controls.setNuInfo(nu / (du * (1 / e.n)), hc ? hc.floor : NaN, nu, this.P.me);
    }
    const g = (o) => { const v = e.peek(['dsc', o]); return v === undefined ? NaN : v; };
    const dsc = g(this.P.order), dsc1 = g(1);
    this.controls.setDsc(dsc, dsc1, this.P.D);
    this.controls.setNuEff(e.hcCached(this.P), this.P.nu);
    this.spectra.updateReadouts(this.P, { dsc, dsc1 });
    if (this.legendre && this.visible('l')) this.legendre.updateReadouts(this.P);
  }

  hcIfCached(P) { return this.eng.hcCached(P); }

  showTimings() {
    const parts = [];
    for (const [k, v] of this.eng.timings) parts.push(`${k} ${fmtMs(v)}`);
    this.timingEl.textContent = parts.join(' · ');
    this.timingEl.title = parts.join('\n');
  }
}
