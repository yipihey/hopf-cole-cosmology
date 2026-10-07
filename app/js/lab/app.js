// app.js - orchestrates state, engine, controls, field panels and spectra.
//
// Computation model: every user-visible computation is a *task* in a chain
// (see run()). Between tasks control returns to the browser (so the status
// line and the already finished panels repaint), and a newer run() cancels the
// older chain at the next task boundary. Results are memoised in the Engine,
// so toggling panels or series never recomputes anything that is cached.

import { getGPU, gpuStatus, onGpuEvent } from '../viz/gpu.js';
import { Engine } from './engine.js';
import { decodeHash, encodeHash, defaultN, defaultLive, makeDefaults, ENV } from './state.js';
import { queryFlags, decidePerf, runBench, diagnosticsText } from './perf.js';
import { buildControls } from './controls.js';
import { Fields2D } from './fields2d.js';
import { Fields3D } from './fields3d.js';
import { Spectra } from './spectra.js';
import { LegendreLab } from './legendre.js';
import { buildExplain } from './explain.js';
import { el, tick, paint, fmtMs } from './dom.js';

const $ = (id) => document.getElementById(id);

const GPU_ERR_RE = /WebGPU|GPU|device lost|out of memory|allocation/i;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export class Lab {
  /** The WASM core is attached later (start() receives its promise), so the page can render before it has loaded. */
  constructor(core = null) {
    this.core = core;
    this.flags = queryFlags();
    const hq = new URLSearchParams(location.hash.replace(/^#/, ''));
    // keys given explicitly in the URL hash are never overridden by the performance presets
    this.explicit = new Set(['n', 'gc', 'lv', 'f'].filter((k) => hq.has(k)));
    this.lite = decidePerf(this.flags.lite ? 'lite' : (hq.get('perf') || 'auto'), this.flags, null, null).lite;   // provisional: sync facts only
    ENV.lite = this.lite;
    this.S = decodeHash(location.hash);
    if (this.flags.lite) this.S.perf = 'lite';
    this.P = structuredClone(this.S);
    this.eng = new Engine(core);
    this.runGen = 0;
    this.timers = {};
    this.dirty = false;
    this.fields = null; this.fieldsMode = 0; this.fieldsN = 0;
    this.spectra = null; this.legendre = null;
    this.gpu = false; this.gpuCompute = false; this.backend = 'canvas2d';
    this.bench = null; this.decision = null; this.firstRunDone = false; this.lastError = null; this.wasmExports = null;
    window.addEventListener('hashchange', () => { if (location.hash.replace(/^#/, '') !== encodeHash(this.S)) location.reload(); });
  }

  /**
   * Start-up order: controls, status line and panels first (the page is usable at once), then the WASM core and the GPU probe in
   * parallel (the probe is capped at 4 s and never rejects), then the performance decision, then the first run under a 20 s watchdog.
   */
  async start(corePromise = null) {
    this.statusEl = $('lab-status-text');
    this.timingEl = $('lab-status-timing');
    this.t0 = performance.now();
    if (new URLSearchParams(location.search).has('embed')) document.body.classList.add('embed');
    this.setStatus('building controls …', 'busy');
    this.setBackendBadge('probing …');
    if (!this.explicit.has('lv')) { this.S.live = defaultLive(this.S.mode, this.S.gc); this.P.live = this.S.live; }
    this.controls = buildControls(this, $('lab-controls'));
    this.buildVisChips();
    this.spectra = new Spectra(this, $('spectra-host'));
    this.legendre = new LegendreLab(this, $('legendre-host'));
    buildExplain($('explain-host'));
    this.applyVisibility();
    this.ensureFields(this.S);
    window.__lab = this;       // debugging / tests
    this.controls.setPerfNote('performance: probing …');
    this.updateDiagnostics();
    this.tControls = performance.now() - this.t0;
    await paint();

    // WASM core and GPU probe in parallel, with a progress text
    let coreDone = !corePromise, gpuDone = false;
    const progress = () => this.setStatus(`${coreDone ? '' : 'loading WebAssembly core … '}${gpuDone ? '' : (coreDone ? '' : '· ') + 'probing GPU …'}`.trim(), 'busy');
    progress();
    const coreP = corePromise ? corePromise.then((c) => { coreDone = true; progress(); return c; }) : Promise.resolve(this.core);
    const gpuP = this.probeGpu().then(() => { gpuDone = true; progress(); });
    const core = await coreP;                   // a load failure propagates to main.js (fatal)
    await gpuP;
    this.core = core; this.eng.core = core;
    try { this.wasmExports = await core.default(); } catch (e) { this.wasmExports = null; }   // already initialised: returns the exports

    // performance decision
    const dec = decidePerf(this.S.perf, this.flags, gpuStatus(), this.bench);
    this.applyPerf(dec);
    this.controls.syncAll();
    if (this.fields.syncBackend) this.fields.syncBackend();
    this.tProbe = performance.now() - this.t0;
    const gst = gpuStatus();
    if (gst.trouble) this.showBanner(`WebGPU could not be initialised (${gst.reason}); continuing on Canvas2D / WASM.`);
    this.setStatus(`performance: ${this.lite ? 'lite' : 'full'} (${dec.reason}) · first run …`, 'busy');
    this.updateDiagnostics();
    await paint();

    this.watchdog = setTimeout(() => {
      if (!this.firstRunDone) this.showBanner('The first run has not finished within 20 s.');
    }, 20000);
    try { await this.run(); } catch (e) { this.reportError('first run', e); this.showBanner('The first run failed: ' + ((e && e.message) || e)); }
    this.tFirstRun = performance.now() - this.t0;
    this.updateDiagnostics();
  }

  // ---------------------------------------------------------------- GPU probe, performance presets, diagnostics

  /** Acquire the shared GPU (4 s cap, never rejects) and run the micro-benchmark. */
  async probeGpu() {
    let gpu = null;
    try { gpu = await getGPU(); } catch (e) { gpu = null; }
    this.gpuObj = gpu;
    this.gpu = !!gpu;
    this.backend = this.gpu ? 'webgpu' : 'canvas2d';
    // WebGPU compute path (3D): needs the shared device; default 'live update' depends on it
    this.gpuCompute = !!gpu;
    ENV.gpuCompute = this.gpuCompute;
    this.eng.gpuDev = gpu ? gpu.device : null;
    if (gpu) {
      this.offGpu = onGpuEvent((ev) => this.onGpuEvent(ev));
      this.bench = await runBench(gpu.device, this.flags);
    }
    if (!this.explicit.has('lv')) { this.S.live = defaultLive(this.S.mode, this.S.gc); this.P.live = this.S.live; }
    this.setBackendBadge();
    return gpu;
  }

  setBackendBadge(text) {
    const b = $('lab-status-backend');
    if (!b) return;
    b.textContent = text ? `backend: ${text}` : `backend: ${this.backend} · ${this.lite ? 'lite' : 'full'}`;
    b.title = this.decision ? `performance preset ${this.lite ? 'lite' : 'full'}: ${this.decision.reason}` : 'graphics backend and performance preset';
  }

  /**
   * Make `dec` the active preset. Re-applies the lite/full defaults (2D grid, GPU compute, 2D sheet panel, live update, 3D volume panel)
   * except for values given explicitly in the URL hash (or set by hand since).
   */
  applyPerf(dec) {
    const S = this.S, was = this.lite;
    const old = makeDefaults(S.mode);                   // defaults of the previous preset
    this.lite = dec.lite; this.decision = dec; ENV.lite = dec.lite;
    const nu = makeDefaults(S.mode);
    if (was !== this.lite) {
      if (!this.explicit.has('n') && S.n === old.n) S.n = nu.n;
      if (!this.explicit.has('gc')) S.gc = nu.gc;
      if (!this.explicit.has('f')) S.slots = S.slots.map((q, i) => (same(q, old.slots[i]) ? nu.slots[i] : q));
    }
    if (!this.explicit.has('lv')) S.live = defaultLive(S.mode, S.gc);
    if (this.controls) {
      this.controls.setPerfNote(`${S.perf} → ${this.lite ? 'lite' : 'full'}: ${dec.reason}`);
      if (this.fields && this.fields.syncPerf) this.fields.syncPerf();
    }
    this.setBackendBadge();
    this.hashChanged();
    return was !== this.lite;
  }

  /** The Performance select. A user choice overrides ?lite=1. */
  setPerf(pref) {
    const S = this.S;
    S.perf = pref;
    const dec = decidePerf(pref, { ...this.flags, lite: false }, gpuStatus(), this.bench);
    const changed = this.applyPerf(dec);
    this.controls.syncAll();
    this.updateDiagnostics();
    if (!changed) { this.setStatus(`performance: ${this.lite ? 'lite' : 'full'} (${dec.reason})`); return; }
    this.eng.releaseGpu();
    this.invalidate();
    this.appliedKey = null;
    this.markStale(true);
    this.ensureFields(S);
    this.controls.syncVisibility();
    if (S.live) { this.setStatus(`performance: ${this.lite ? 'lite' : 'full'} (${dec.reason}) · recomputing …`, 'busy'); this.schedule(30); }
    else { this.setDirty(true); this.setStatus(`performance: ${this.lite ? 'lite' : 'full'} (${dec.reason}); press Run to apply the new defaults.`); }
  }

  /** Debounce: lite waits at least 250 ms after the last change. */
  lag(ms) { return this.lite ? Math.max(ms, 250) : ms; }

  updateDiagnostics() {
    if (!this.controls || !this.controls.setDiag) return;
    const mem = this.wasmExports && this.wasmExports.memory ? this.wasmExports.memory.buffer.byteLength : null;
    const t = [];
    if (this.tControls !== undefined) t.push(`controls ${this.tControls.toFixed(0)} ms`);
    if (this.tProbe !== undefined) t.push(`WASM + GPU probe done ${this.tProbe.toFixed(0)} ms`);
    if (this.tFirstRun !== undefined) t.push(`first run done ${this.tFirstRun.toFixed(0)} ms`);
    this.controls.setDiag(diagnosticsText({
      gpu: gpuStatus(), bench: this.bench, decision: this.decision, backend: this.backend, lite: this.lite, pref: this.S.perf,
      wasmBytes: mem, lastError: this.lastError || this.lastErrorSticky, startupMs: t.join(', '),
    }));
  }

  // ---------------------------------------------------------------- trouble banner and recovery

  /** Non-blocking banner at the top of the main area. */
  showBanner(detail) {
    const b = $('lab-banner');
    if (!b) return;
    b.textContent = '';
    el('span', 'lab-banner-msg', b, 'The lab is having trouble with this GPU.');
    if (!this.lite) {
      const a = el('a', null, b, '[Switch to lite mode]'); a.href = '#';
      a.addEventListener('click', (ev) => { ev.preventDefault(); this.hideBanner(); this.controls.perf.set('lite'); this.setPerf('lite'); });
    }
    if (this.gpu) {
      const r = el('a', null, b, '[Reload without WebGPU]');
      const u = new URL(location.href); u.searchParams.set('nogpu', '1'); u.searchParams.delete('gpufail');
      const h = encodeHash(this.S); u.hash = h ? '#' + h : '';
      r.href = u.toString();
    }
    const x = el('button', 'lab-banner-x', b, '×'); x.type = 'button'; x.title = 'dismiss'; x.addEventListener('click', () => this.hideBanner());
    if (detail) el('span', 'lab-banner-detail', b, detail);
    b.hidden = false;
    this.lastErrorSticky = detail;
    this.updateDiagnostics();
  }
  hideBanner() { const b = $('lab-banner'); if (b) b.hidden = true; }

  onGpuEvent(ev) {
    if (ev.type === 'lost') { this.handleDeviceLost(ev); return; }
    if (ev.type === 'error') {
      this.updateDiagnostics();
      if (/out of memory|OutOfMemory|OOM|allocation/i.test(`${ev.kind} ${ev.message}`)) this.showBanner('GPU error: ' + ev.message);
    }
  }

  /** The GPU device was lost: continue on Canvas2D / WASM immediately, without throwing. */
  handleDeviceLost(ev) {
    console.warn('[lab] GPU device lost; falling back to Canvas2D/WASM:', ev.message);
    this.gpu = false; this.gpuCompute = false; ENV.gpuCompute = false; this.backend = 'canvas2d';
    this.eng.gpuDev = null;
    try { this.eng.releaseGpu(); } catch (e) { /* the device is gone */ }
    this.eng.gpuWanted = false;
    const dec = decidePerf(this.S.perf, { ...this.flags, lite: false }, gpuStatus(), this.bench);
    this.applyPerf(dec);
    // the views hold the dead device: rebuild the panels (their canvases get the Canvas2D fallbacks)
    try {
      if (this.fields) { this.fields.destroy(); this.fields = null; this.fieldsMode = 0; }
      if (this.legendre) { this.legendre.destroy(); this.legendre = new LegendreLab(this, $('legendre-host')); }
      this.ensureFields(this.S);
    } catch (e) { console.error(e); }
    this.setBackendBadge();
    this.controls.syncAll();
    this.showBanner('GPU device lost (' + ev.message + '); continuing on Canvas2D / WASM.');
    this.setStatus('GPU device lost: continuing on Canvas2D / WASM', 'err');
    this.invalidate();
    this.appliedKey = null;
    this.markStale(true);
    if (this.eng.sim) {
      if (this.S.live) this.schedule(100); else this.setDirty(true);
    }
  }

  /** Real (not memory-refusal) failure of the GPU compute path: it has fallen back to WASM; switch the checkbox off. */
  onGpuComputeFailed() {
    const S = this.S;
    if (S.gc) { S.gc = false; S.live = defaultLive(S.mode, false); this.hashChanged(); this.controls.syncAll(); }
    this.showBanner('GPU compute path failed: ' + this.eng.gpuError);
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
    if (GPU_ERR_RE.test(String(err && err.message ? err.message : err))) this.showBanner(this.lastError);
    else this.updateDiagnostics();
  }

  /** Called by controls for every changed parameter. */
  setParam(key, value, level) {
    const S = this.S;
    S[key] = value;
    if (key === 'n' || key === 'live') this.explicit.add(key === 'n' ? 'n' : 'lv');   // set by hand: presets leave it alone
    this.hashChanged();
    if (level === 'live') {
      if (value) this.schedule(this.lag(60));
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
      this.schedule(this.lag(170));
    } else {
      this.markStale(true);
      this.schedule(this.lag(level === 'ic' ? 70 : 110));
    }
  }

  /** The "GPU compute" checkbox (3D). Live update follows its default; the other path recomputes. */
  setGpuCompute(v) {
    const S = this.S;
    S.gc = v;
    this.explicit.add('gc');
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
    this.timers.hc = setTimeout(() => this.runHc(), this.lag(this.S.mode === 3 ? 40 : 80));
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
      const tt = performance.now();
      try {
        await t.fn();
        (this.taskLog || (this.taskLog = [])).push([t.label, Math.round(performance.now() - tt)]); if (this.taskLog.length > 200) this.taskLog.shift();
      } catch (err) {
        this.reportError(t.label, err);
        if (t.critical) { document.body.classList.remove('lab-working'); return false; }
      }
    }
    document.body.classList.remove('lab-working');
    return true;
  }

  /** Full (re)computation for the current control state. Never throws: unexpected errors are reported (and GPU-related ones raise the banner). */
  async run() {
    try { await this._run(); }
    catch (err) {
      this.reportError('run', err);
      document.body.classList.remove('lab-working');
      this.firstRunDone = true;
    }
  }

  async _run() {
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
    if (!this.firstRunDone) {
      this.firstRunDone = true; clearTimeout(this.watchdog);
      if (!ok && this.lastError) this.showBanner('The first run failed: ' + this.lastError);
    }
    this.refreshReadouts();
    if (P.mode === 3 && this.eng.gpuError && !this.eng.gpuRefused) this.onGpuComputeFailed();
    if (ok && !this.lastError) this.appliedKey = this.coreKey(P);
    if (this.lastError) this.setStatus(this.lastError, 'err');
    else if (ok) {
      const path = this.eng.pathLabel(P);
      this.setStatus(`ready · ${P.mode}D ${P.n}${P.mode === 2 ? '²' : '³'} · order ${P.order} · D = ${P.D.toFixed(3)}${path ? ' · ' + path : ''} · ${fmtMs(performance.now() - t0)}`, this.eng.gpuError && P.mode === 3 ? 'err' : '');
    }
    this.showTimings();
    this.updateDiagnostics();
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
