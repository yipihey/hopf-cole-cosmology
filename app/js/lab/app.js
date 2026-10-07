// app.js - orchestrates state, engine, controls, field panels and spectra.
//
// Computation model: every user-visible computation is a *task* in a chain
// (see run()). Between tasks control returns to the browser (so the status
// line and the already finished panels repaint), and a newer run() cancels the
// older chain at the next task boundary. Results are memoised in the Engine,
// so toggling panels or series never recomputes anything that is cached.

import { getGPU } from '../viz/gpu.js';
import { Engine } from './engine.js';
import { decodeHash, encodeHash, defaultN, defaultLive } from './state.js';
import { buildControls } from './controls.js';
import { Fields2D } from './fields2d.js';
import { Fields3D } from './fields3d.js';
import { Spectra } from './spectra.js';
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
    this.spectra = null;
    window.addEventListener('hashchange', () => { if (location.hash.replace(/^#/, '') !== encodeHash(this.S)) location.reload(); });
  }

  async start() {
    const gpu = await getGPU();
    this.gpu = !!gpu;
    this.backend = this.gpu ? 'webgpu' : 'canvas2d';
    if (new URLSearchParams(location.search).has('embed')) document.body.classList.add('embed');
    this.statusEl = $('lab-status-text');
    this.timingEl = $('lab-status-timing');
    $('lab-status-backend').textContent = `backend: ${this.backend}`;

    this.controls = buildControls(this, $('lab-controls'));
    this.buildVisChips();
    this.spectra = new Spectra(this, $('spectra-host'));
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
    this.invalidate();
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

  setMode(m) {
    if (m === this.S.mode) return;
    const S = this.S;
    S.mode = m; S.n = defaultN(m); S.live = defaultLive(m);
    if (m === 3 && S.me > 1) S.me = 1;
    this.invalidate();
    this.controls.syncAll();
    this.spectra.syncMode();
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
  }

  // ---------------------------------------------------------------- sections

  buildVisChips() {
    const host = $('lab-vis');
    const defs = [['f', 'Fields'], ['s', 'Spectra'], ['e', 'Explain']];
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
        if (cb.checked && this.eng.sim && (k === 'f' || k === 's')) this.runSection(k);
      });
    }
  }
  applyVisibility() {
    const v = this.S.vis;
    $('sec-fields').hidden = !v.includes('f');
    $('sec-spectra').hidden = !v.includes('s');
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
    tasks.push({ label: 'initial conditions', heavy: icChanged, critical: true, fn: () => { this.eng.configure(P); } });
    tasks.push({ label: `LPT order ${P.order}`, heavy: true, critical: true, fn: () => { this.eng.ensureLpt(P.order); } });
    if (this.visible('f')) {
      const ft = this.fields.tasks(P);
      tasks.push(...ft.filter((t) => !t.heavy), ...ft.filter((t) => t.heavy));
    }
    if (this.visible('s')) tasks.push(...this.spectra.tasks(P));
    tasks.push({ label: 'shell-crossing times', heavy: heavy3, fn: () => { this.eng.dsc(P.order); this.eng.dsc(1); this.refreshReadouts(); } });

    const ok = await this.runTasks(tasks, gen);
    if (gen !== this.runGen) return;
    this.refreshReadouts();
    if (this.lastError) this.setStatus(this.lastError, 'err');
    else if (ok) this.setStatus(`ready · ${P.mode}D ${P.n}${P.mode === 2 ? '²' : '³'} · order ${P.order} · D = ${P.D.toFixed(3)} · ${fmtMs(performance.now() - t0)}`);
    this.showTimings();
  }

  /** Compute only the tasks of one section (when it is switched on). */
  async runSection(k) {
    const gen = this.runGen;
    const P = this.P;
    const tasks = k === 'f' ? this.fields.tasks(P) : this.spectra.tasks(P);
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

  refreshReadouts() {
    const e = this.eng;
    if (!e.sim || !this.controls) return;
    const g = (o) => { const v = e.peek(['dsc', o]); return v === undefined ? NaN : v; };
    const dsc = g(this.P.order), dsc1 = g(1);
    this.controls.setDsc(dsc, dsc1, this.P.D);
    this.controls.setNuEff(e.hcCached(this.P), this.P.nu);
    this.spectra.updateReadouts(this.P, { dsc, dsc1 });
  }

  hcIfCached(P) { return this.eng.hcCached(P); }

  showTimings() {
    const parts = [];
    for (const [k, v] of this.eng.timings) parts.push(`${k} ${fmtMs(v)}`);
    this.timingEl.textContent = parts.join(' · ');
    this.timingEl.title = parts.join('\n');
  }
}
