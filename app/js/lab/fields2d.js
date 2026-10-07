// fields2d.js - the "Fields" section in 2D: four panels with selectable content.

import { FieldView } from '../viz/fieldview.js';
import { SheetView } from '../viz/sheetview.js';
import { renderColorbar, COLORMAPS } from '../viz/colormaps.js';
import { slider, checkbox } from '../viz/ui.js';
import { transpose2D } from '../hcc.js';
import { el, sel, fmtNum } from './dom.js';
import { CATALOG, KIND_ORDER, LIN_MAX, psihatRange } from './catalog.js';

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
    return { kind, def, sub, cmap, log };
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
    this.opts = el('div', 'lab-popts', root);
    this.subHost = el('span', 'lab-subhost', this.opts);
    this.cmapSel = sel(this.opts, { label: 'map', options: CMAPS, onChange: (v) => { this.save({ cmap: v }); this.redraw(); } });
    this.logCb = checkbox(this.opts, { label: 'log', value: true, onChange: (v) => { this.save({ log: v ? '1' : '0' }); this.redraw(); } });
    const wb = el('div', 'hcc-with-bar', root);
    const stage = this.stage = el('div', 'hcc-stage', wb);
    this.cvF = el('canvas', 'lab-cv', stage);
    this.cvS = el('canvas', 'lab-cv', stage);
    this.cvS.hidden = true;
    this.bar = el('canvas', 'hcc-colorbar', wb); this.bar.width = 84; this.bar.height = 8;
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
    this.cmapSel.set(cmap);
    this.logCb.set(log);
    this.logCb.el.hidden = def.cls !== 'density';
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
      this.cap.textContent = (def.gpu && !this.app.gpu ? CATALOG.sheetcpu : def).caption(sub);
      this.note.hidden = true;
      const n = eng.n;
      if (def.gpu && this.app.gpu) {
        await this.drawSheetGPU(P, sub, cmap, log);
        return;
      }
      this.cvS.hidden = true; this.cvF.hidden = false;
      let drawDef = def;
      if (def.gpu) {   // no WebGPU: fall back to the CPU-rasterized sheet
        drawDef = CATALOG.sheetcpu;
        this.note.hidden = false;
        this.note.textContent = 'WebGPU unavailable: showing the CPU-rasterized sheet density instead.';
      }
      const fv = await this.ensureFV();
      const arr = drawDef.data(eng, P, sub);
      if (!arr || arr.length < n * n) throw new Error(`panel data has ${arr ? arr.length : 0} values, expected ${n * n}`);
      fv.setField(transpose2D(arr, n), n, n);
      fv.setColormap(cmap);
      switch (def.cls) {
        case 'density': { const [a, b] = this.densityRange(log); fv.setRange(a, b, { log }); break; }
        case 'sym': fv.setRange(undefined, undefined, { symmetric: true }); break;
        case 'amp': fv.setRange(-6, 0); break;
        case 'phase': fv.setRange(-Math.PI, Math.PI); break;
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
    sv.setMesh(eng.positions(P), eng.n, 1.0);
    this.drawSheetParams(mode, cmap, log);
    this.markStale(false);
  }

  drawSheetParams(mode, cmap, log) {
    const S = this.app.S;
    const lo = log ? S.rmin : 0, hi = log ? S.rmax : LIN_MAX;
    this.sv.draw({ mode, cmap, vmin: lo, vmax: hi, log, wireAlpha: Math.min(0.35, 25 / this.app.eng.n) });
    renderColorbar(this.bar, cmap, lo, hi, { label: 'ρ/ρ̄', log });
  }

  /** Cheap redraw from cached data after a display-only change. */
  redraw() { this.app.runPanel(this.owner, this.idx); }

  /** D slider fast path: only the GPU sheet panel (positions + draw). */
  fastUpdate(P) {
    if (!this.isFast() || !this.sv || this.cvS.hidden || !this.app.eng.sim) return false;
    const { sub, cmap, log } = this.cfg;
    this.sv.setMesh(this.app.eng.positions(P), this.app.eng.n, 1.0);
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
    const onR = () => { app.hashChanged(); this.redrawAll(); };
    this.rmin = slider(bar, { label: 'ρ min', min: 0.01, max: 1, value: app.S.rmin, log: true, onInput: (v) => { app.S.rmin = v; onR(); } });
    this.rmax = slider(bar, { label: 'ρ max', min: 3, max: 1000, value: app.S.rmax, log: true, onInput: (v) => { app.S.rmax = v; onR(); } });
    this.grid = el('div', 'hcc-grid2 lab-grid', this.root);
    this.slots = [0, 1, 2, 3].map((i) => new Slot(this, i, this.grid));
    this.ro = new ResizeObserver(() => this.slots.forEach((s) => s.resizeBar()));
    this.ro.observe(this.grid);
  }
  redrawAll() { this.slots.forEach((s, i) => this.app.runPanel(this, i)); }
  markStale(b, keepFast = false) { this.slots.forEach((s) => s.markStale(b && !(keepFast && s.isFast()))); }
  fastUpdate(P) { let any = false; for (const s of this.slots) any = s.fastUpdate(P) || any; return any; }
  tasks(P) {
    return this.slots.map((s, i) => ({
      label: `panel ${i + 1}: ${CATALOG[s.cfg.kind].label}`,
      heavy: !s.isFast(),
      fn: () => s.update(P),
    }));
  }
  destroy() { this.ro.disconnect(); this.slots.forEach((s) => s.destroy()); this.root.remove(); }
}
