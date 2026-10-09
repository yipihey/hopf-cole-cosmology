// fields2d.js - the "Fields" section in 2D: four panels with selectable content.

import { FieldView } from '../viz/fieldview.js';
import { SheetView } from '../viz/sheetview.js';
import { renderColorbar, COLORMAPS } from '../viz/colormaps.js';
import { slider, checkbox } from '../viz/ui.js';
import { transpose2D } from '../hcc.js';
import { el, sel, fmtNum, setCaption } from './dom.js';
import { pngButton, exportPNG, pngName } from './export.js';
import { CATALOG, KIND_ORDER, LIN_MAX, psihatRange } from './catalog.js';
import { Engine } from './engine.js';

const CMAPS = Object.keys(COLORMAPS).map((c) => [c, c]);
const KIND_OPTIONS = KIND_ORDER.map((k) => [k, CATALOG[k].label]);

class Slot {
  constructor(owner, idx, host) {
    this.owner = owner; this.idx = idx;
    this.fv = null; this.sv = null;
    this.build(host);
  }
  get app() { return this.owner.app; }
  get cfg() {
    const c = this.app.S.slots[this.idx];
    const kind = c[0];
    const def = CATALOG[kind];
    const sub = c[1] || (def.subs ? def.subs.def : '');
    const cmap = c[2] || def.cmap;
    const log = c[3] === '' ? !!def.log : c[3] === '1';
    const opt = c[3] !== '' && def.opt && def.opt.options.some((o) => o[0] === c[3]) ? c[3] : (def.opt ? def.opt.def : '');   // kinds with a per-panel option keep it in the 4th slot
    return { kind, def, sub, cmap, log, opt };
  }
  save(patch) {
    const c = this.app.S.slots[this.idx];
    const kind = patch.kind !== undefined ? patch.kind : c[0];
    this.app.S.slots[this.idx] = [kind, patch.sub !== undefined ? patch.sub : c[1], patch.cmap !== undefined ? patch.cmap : c[2], patch.log !== undefined ? patch.log : c[3]];
    this.app.hashChanged();
  }

  build(host) {
    const root = this.root = el('div', 'hcc-panel lab-panel', host);
    const head = el('div', 'lab-ph', root);
    this.kindSel = sel(head, { options: KIND_OPTIONS, value: this.app.S.slots[this.idx][0], onChange: (v) => {
      this.save({ kind: v, sub: '', cmap: '', log: '' });
      this.syncControls();
      this.app.runPanel(this.owner, this.idx);
    } });
    this.kindSel.el.classList.add('lab-kind');
    this.pngBtn = pngButton(head, (b) => this.exportPNG(b));
    this.opts = el('div', 'lab-popts', root);
    this.subHost = el('span', 'lab-subhost', this.opts);
    this.cmapSel = sel(this.opts, { label: 'map', options: CMAPS, onChange: (v) => { this.save({ cmap: v }); this.redraw(); } });
    this.optHost = el('span', 'lab-subhost', this.opts);
    this.logCb = checkbox(this.opts, { label: 'log', value: true, onChange: (v) => { this.save({ log: v ? '1' : '0' }); this.redraw(); } });
    const wb = el('div', 'hcc-with-bar', root);
    const stage = this.stage = el('div', 'hcc-stage', wb);
    this.cvF = el('canvas', 'lab-cv', stage);
    this.cvS = el('canvas', 'lab-cv', stage);
    this.cvS.hidden = true;
    this.bar = el('canvas', 'hcc-colorbar', wb); this.bar.width = 84; this.bar.height = 8;
    this.cvF.setAttribute('role', 'img'); this.cvS.setAttribute('role', 'img');
    this.bar.setAttribute('role', 'img');
    this.hover = el('div', 'lab-hover hcc-note', root, ' ');
    this.cap = el('p', 'hcc-note lab-cap', root);
    this.note = el('p', 'lab-warn-note', root); this.note.hidden = true;
    this.cvF.addEventListener('pointermove', (ev) => this.onHover(ev));
    this.cvF.addEventListener('pointerleave', () => { this.hover.textContent = ' '; });
    this.syncControls();
  }

  /** Reflect cfg into the select/checkbox widgets (and rebuild the sub-selector). */
  syncControls() {
    const { kind, def, sub, cmap, log } = this.cfg;
    this.kindSel.set(kind);
    this.updateAria();
    this.cmapSel.set(cmap);
    this.logCb.set(log);
    this.logCb.el.hidden = def.cls !== 'density';
    this.optHost.textContent = '';
    if (def.opt) {
      sel(this.optHost, { label: def.opt.label, options: def.opt.options, value: this.cfg.opt, title: def.opt.title, onChange: (v) => { this.save({ log: v }); this.app.runPanel(this.owner, this.idx); } });
    }
    this.subHost.textContent = '';
    this.subSel = null;
    const gpuless = def.gpu && !this.app.gpu;
    if (def.subs && !gpuless) {
      this.subSel = sel(this.subHost, { label: def.subs.label, options: def.subs.options, value: sub, onChange: (v) => {
        this.save({ sub: v }); this.app.runPanel(this.owner, this.idx);
      } });
    }
  }

  markStale(b) { this.root.classList.toggle('is-stale', b); }

  /** Accessible names of the canvases: panel number, content, selected sub-option. */
  updateAria() {
    const { def, sub } = this.cfg;
    const so = def.subs && def.subs.options.find((o) => o[0] === sub);
    const what = `Panel ${this.idx + 1}: ` + (def.label.includes('…') ? def.label.replace('…', so ? so[1] : '').trim() : `${def.label}${so ? ` (${def.subs.label}: ${so[1]})` : ''}`);
    this.cvF.setAttribute('aria-label', what);
    this.cvS.setAttribute('aria-label', what);
    this.bar.setAttribute('aria-label', `Colour scale of panel ${this.idx + 1}${def.unit ? ': ' + def.unit : ''}`);
    this.pngBtn.setAttribute('aria-label', `Download panel ${this.idx + 1} as PNG`);
  }

  /** PNG of the stage and the colour bar; the visible view is redrawn synchronously first (WebGPU canvases do not keep their buffer). */
  exportPNG(btn) {
    const gpuSheet = !this.cvS.hidden && this.sv;
    const view = gpuSheet ? this.sv : this.fv;
    if (!view) return Promise.resolve(null);
    const { kind, sub } = this.cfg;
    return exportPNG({ canvas: gpuSheet ? this.cvS : this.cvF, bar: this.bar, redraw: () => view.draw() }, pngName(kind, sub, this.app.P.D), btn);
  }

  /** True when the panel shows something derived from the Hopf-Cole solution (recomputed by the fast nu path). */
  dependsOnHc() {
    const { kind, sub } = this.cfg;
    return ['hc', 'hcdual', 'phi', 'lnpsi', 'psihat', 'speed'].includes(kind) || ((kind === 'fabs' || kind === 'fphase' || kind === 'fphaseonly') && (sub === 'hc' || sub === 'hcdual'));
  }

  isFast() { return this.cfg.def.gpu && this.app.gpu; }

  async ensureFV() {
    if (!this.fvPromise) {
      this.fv = new FieldView(this.cvF, { cmap: 'viridis', interpolate: false });
      this.fvPromise = this.fv.init();
    }
    await this.fvPromise;
    return this.fv;
  }
  async ensureSV() {
    if (!this.svPromise) {
      this.sv = new SheetView(this.cvS, { cmap: 'magma' });
      this.svPromise = this.sv.init();
    }
    await this.svPromise;
    return this.sv;
  }

  densityRange(log) {
    const S = this.app.S;
    if (log) return S.same ? [S.rmin, S.rmax] : [undefined, undefined];
    return S.same ? [0, LIN_MAX] : [undefined, undefined];
  }

  /** Draw the panel for applied parameters P. */
  async update(P) {
    const eng = this.app.eng;
    if (!eng.sim) return;
    this.updating = true;
    try {
      let { kind, def, sub, cmap, log } = this.cfg;
      this.syncHidden(def);
      const opt = this.cfg.opt;
      this.updateAria();
      setCaption(this.cap, (def.gpu && !this.app.gpu ? CATALOG[def.cpu] : def).caption(sub, P, eng, opt), def);
      this.note.hidden = true;
      const n = eng.n;
      if (def.gpu && this.app.gpu) {
        await this.drawSheetGPU(P, sub, cmap, log);
        return;
      }
      this.cvS.hidden = true; this.cvF.hidden = false;
      let drawDef = def;
      if (def.gpu) {   // no WebGPU: fall back to the CPU-rasterized sheet
        drawDef = CATALOG[def.cpu];
        this.note.hidden = false;
        this.note.textContent = 'WebGPU unavailable: showing the CPU-rasterized ' + (def.p1 ? 'P1 ' : '') + 'sheet density instead.';
      }
      const fv = await this.ensureFV();
      // NUFFT density (WASM): unavailable on grids where the refined fine grid would be too large
      if (kind === 'nufft' || ((def.fourier || def.sf) && sub === 'nufft')) {
        const ni = eng.nufftInfo(P);
        if (!ni.ok) { this.note.hidden = false; this.note.textContent = ni.note; this.markStale(false); return; }
      }
      // exact (clipped) sheet deposits come from the GPU clipper when available: make them resident before the synchronous getters run
      const needF = def.need || (((def.fourier || def.sf) && Engine.isExact(sub)) ? sub : null);
      if (needF) await eng.need('field', needF, P);
      const arr = drawDef.data(eng, P, sub, opt);
      if (needF) setCaption(this.cap, def.caption(sub, P, eng, opt), def);
      if (!arr || arr.length < n * n) throw new Error(`panel data has ${arr ? arr.length : 0} values, expected ${n * n}`);
      fv.setField(transpose2D(arr, n), n, n);
      fv.setColormap(cmap);
      switch (def.cls) {
        case 'density': { const [a, b] = this.densityRange(log); fv.setRange(a, b, { log }); break; }
        case 'sym': fv.setRange(undefined, undefined, { symmetric: true }); break;
        case 'amp': fv.setRange(-6, 0); break;
        case 'phase': fv.setRange(-Math.PI, Math.PI); break;
        case 'phaseonly': fv.setRange(-3, 3); break;     // unit-rms map: linear, ±3 rms
        case 'psihat': { const [a, b] = psihatRange(arr); fv.setRange(a, b); break; }
        default: fv.setRange();
      }
      fv.draw();
      const r = fv.getRange();
      renderColorbar(this.bar, cmap, r.vmin, r.vmax, { label: def.unit, log: r.log });
      this.lastKind = def;
      this.markStale(false);
    } finally { this.updating = false; }
  }

  syncHidden(def) {
    // keep widgets in sync if the config was changed programmatically
    if (this.shownKind !== this.cfg.kind) { this.shownKind = this.cfg.kind; this.syncControls(); }
  }

  async drawSheetGPU(P, mode, cmap, log) {
    const eng = this.app.eng;
    const sv = await this.ensureSV();
    this.cvS.hidden = false; this.cvF.hidden = true;
    this.setMeshAndWeights(sv, P);
    this.drawSheetParams(mode, cmap, log);
    this.markStale(false);
  }

  /** Positions at (D, order) and, for the P1 panel, the vertex densities 1/|J| on the Lagrangian grid. */
  setMeshAndWeights(sv, P) {
    const eng = this.app.eng;
    sv.setMesh(eng.positions(P), eng.n, 1.0);
    sv.setVertexWeights(this.cfg.def.p1 ? eng.vertexW(P) : null);
  }

  drawSheetParams(mode, cmap, log) {
    const S = this.app.S;
    const lo = log ? S.rmin : 0, hi = log ? S.rmax : LIN_MAX;
    this.sv.draw({ mode, cmap, vmin: lo, vmax: hi, log, p1: !!this.cfg.def.p1, wireAlpha: Math.min(0.35, 25 / this.app.eng.n) });
    renderColorbar(this.bar, cmap, lo, hi, { label: 'ρ/ρ̄', log });
  }

  /** Cheap redraw from cached data after a display-only change. */
  redraw() { this.app.runPanel(this.owner, this.idx); }

  /** D slider fast path: only the GPU sheet panel (positions + draw). */
  fastUpdate(P) {
    if (!this.isFast() || !this.sv || this.cvS.hidden || !this.app.eng.sim) return false;
    const { sub, cmap, log } = this.cfg;
    this.setMeshAndWeights(this.sv, P);
    this.drawSheetParams(sub, cmap, log);
    return true;
  }

  onHover(ev) {
    const fv = this.fv;
    if (!fv || this.cvF.hidden) return;
    const p = fv.pick(ev);
    if (!p) { this.hover.textContent = ' '; return; }
    const n = fv.m, def = this.cfg.def;
    const where = def.fourier ? `kx=${p.col - (n >> 1)}, ky=${p.row - (n >> 1)} (units of 2π/L)`
      : def.lagr ? `q=(${((p.col + 0.5) / n).toFixed(3)}, ${((p.row + 0.5) / n).toFixed(3)})`
        : `x=(${((p.col + 0.5) / n).toFixed(3)}, ${((p.row + 0.5) / n).toFixed(3)})`;
    this.hover.textContent = `${where}: ${fmtNum(p.value, 4)}`;
  }

  resizeBar() {
    const r = this.lastKind && this.fv && !this.cvF.hidden ? this.fv.getRange() : null;
    if (r) renderColorbar(this.bar, this.cfg.cmap, r.vmin, r.vmax, { label: this.cfg.def.unit, log: r.log });
  }

  destroy() {
    try { if (this.fv) this.fv.destroy(); } catch (e) { /* ignore */ }
    try { if (this.sv) this.sv.destroy(); } catch (e) { /* ignore */ }
    this.root.remove();
  }
}

export class Fields2D {
  constructor(app, host) {
    this.app = app;
    this.root = el('div', 'lab-fields2d', host);
    const bar = el('div', 'hcc-controls lab-fbar', this.root);
    this.sameCb = checkbox(bar, { label: 'same color range (all density panels)', value: app.S.same, onChange: (v) => { app.S.same = v; app.hashChanged(); this.redrawAll(); } });
    const onR = () => { app.hashChanged(); this.redrawAll(); if (app.legendre) app.legendre.redraw(); };
    this.rmin = slider(bar, { label: 'ρ min', min: 0.01, max: 1, value: app.S.rmin, log: true, onInput: (v) => { app.S.rmin = v; onR(); } });
    this.rmax = slider(bar, { label: 'ρ max', min: 3, max: 1000, value: app.S.rmax, log: true, onInput: (v) => { app.S.rmax = v; onR(); } });
    this.grid = el('div', 'hcc-grid2 lab-grid', this.root);
    this.slots = [0, 1, 2, 3].map((i) => new Slot(this, i, this.grid));
    this.ro = new ResizeObserver(() => this.slots.forEach((s) => s.resizeBar()));
    this.ro.observe(this.grid);
  }
  /** Reflect S.rmin / S.rmax / S.same changed elsewhere (Legendre lab). */
  syncRange() { this.rmin.set(this.app.S.rmin); this.rmax.set(this.app.S.rmax); this.sameCb.set(this.app.S.same); }
  redrawAll() { this.slots.forEach((s, i) => this.app.runPanel(this, i)); }
  /** WebGPU probe finished / device lost: the GPU-only panel kinds change their sub-selectors. */
  syncBackend() { this.slots.forEach((s) => s.syncControls()); }
  syncPerf() { this.slots.forEach((s) => s.syncControls()); }
  markStale(b, keepFast = false) { this.slots.forEach((s) => s.markStale(b && !(keepFast && s.isFast()))); }
  fastUpdate(P) { let any = false; for (const s of this.slots) any = s.fastUpdate(P) || any; return any; }
  tasks(P, hcOnly = false) {
    return this.slots.map((s, i) => ({
      label: `panel ${i + 1}: ${CATALOG[s.cfg.kind].label}`,
      heavy: !s.isFast() && !hcOnly,
      fn: () => s.update(P),
      hc: s.dependsOnHc(),
    })).filter((t) => !hcOnly || t.hc);
  }
  markStaleHc(b) { this.slots.forEach((s) => s.markStale(b && s.dependsOnHc())); }
  destroy() { this.ro.disconnect(); this.slots.forEach((s) => s.destroy()); this.root.remove(); }
}
