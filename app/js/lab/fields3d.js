// fields3d.js - the "Fields" section in 3D: volume rendering, a slice and the k_z = 0 Fourier plane.

import { FieldView } from '../viz/fieldview.js';
import { VolumeView } from '../viz/volumeview.js';
import { renderColorbar, COLORMAPS } from '../viz/colormaps.js';
import { slider, checkbox } from '../viz/ui.js';
import { slice3D } from '../hcc.js';
import { el, sel } from './dom.js';

const CMAPS = Object.keys(COLORMAPS).map((c) => [c, c]);
const FIELDS = [['cic', 'CIC density'], ['hc', 'Hopf–Cole density'], ['lin', 'Linear density 1+Dδ0']];
const FNAME = { cic: 'CIC density', hc: 'Hopf–Cole density 1+δ', lin: 'linear density 1+Dδ0 (clipped at 10⁻³)' };

export class Fields3D {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-fields3d', host);
    const bar = el('div', 'hcc-controls lab-fbar', this.root);
    checkbox(bar, { label: 'slice uses the volume’s colour range', value: S.same, onChange: (v) => { S.same = v; app.hashChanged(); this.updateSlice(); } });
    this.grid = el('div', 'hcc-grid2 lab-grid', this.root);
    this.buildVolume();
    this.buildSlice();
    this.buildFourier(2, 'fabs');
    this.buildFourier(3, 'fphase');
    this.ro = new ResizeObserver(() => this.renderBars());
    this.ro.observe(this.grid);
  }

  panel(title) {
    const root = el('div', 'hcc-panel lab-panel', this.grid);
    const head = el('div', 'lab-ph', root);
    return { root, head, opts: el('div', 'lab-popts', root), title };
  }
  canvasBlock(root, cls) {
    const wb = el('div', 'hcc-with-bar', root);
    const stage = el('div', 'hcc-stage', wb);
    const cv = el('canvas', 'lab-cv' + (cls ? ' ' + cls : ''), stage);
    const bar = el('canvas', 'hcc-colorbar', wb); bar.width = 84; bar.height = 8;
    return { cv, bar };
  }

  buildVolume() {
    const S = this.app.S, app = this.app;
    const p = this.pv = this.panel('Volume');
    p.root.classList.add('is-volume');
    sel(p.head, { options: FIELDS, value: S.v1, onChange: (v) => { S.v1 = v; app.hashChanged(); app.runPanel(this, 0); } });
    sel(p.opts, { label: 'mode', options: [['emission', 'emission'], ['mip', 'max. intensity']], value: S.vm, onChange: (v) => { S.vm = v; app.hashChanged(); this.drawVolume(); } });
    slider(p.opts, { label: 'opacity', min: 0.5, max: 100, value: S.vo, log: true, onInput: (v) => { S.vo = v; app.hashChanged(); this.drawVolume(); } });
    sel(p.opts, { label: 'map', options: CMAPS, value: S.c1, onChange: (v) => { S.c1 = v; app.hashChanged(); this.drawVolume(); } });
    checkbox(p.opts, { label: 'log', value: S.l1, onChange: (v) => { S.l1 = v; app.hashChanged(); app.runPanel(this, 0); } });
    const { cv, bar } = this.canvasBlock(p.root);
    this.cvV = cv; this.barV = bar;
    if (!app.gpu) cv.classList.add('hcc-wide');     // the Canvas2D fallback shows three slices side by side
    this.capV = el('p', 'hcc-note lab-cap', p.root);
    this.hintV = el('p', 'hcc-note lab-hint', p.root, 'Drag to rotate, wheel to zoom, double-click to reset (WebGPU). Without WebGPU: three central slices.');
  }

  buildSlice() {
    const S = this.app.S, app = this.app;
    const p = this.ps = this.panel('Slice');
    sel(p.head, { options: FIELDS, value: S.s1, onChange: (v) => { S.s1 = v; app.hashChanged(); app.runPanel(this, 1); } });
    sel(p.opts, { label: 'axis ⟂', options: [[0, 'x'], [1, 'y'], [2, 'z']], value: S.sa, onChange: (v) => { S.sa = Number(v); app.hashChanged(); this.updateSlice(); } });
    this.idxSl = slider(p.opts, { label: 'index', min: 0, max: Math.max(1, app.S.n - 1), step: 1, value: Math.round(S.si * (S.n - 1)), format: (v) => String(Math.round(v)),
      onInput: (v) => { S.si = v / (S.n - 1); app.hashChanged(); this.updateSlice(); } });
    sel(p.opts, { label: 'map', options: CMAPS, value: S.c2, onChange: (v) => { S.c2 = v; app.hashChanged(); this.updateSlice(); } });
    checkbox(p.opts, { label: 'log', value: S.l2, onChange: (v) => { S.l2 = v; app.hashChanged(); this.updateSlice(); } });
    const { cv, bar } = this.canvasBlock(p.root);
    this.cvS = cv; this.barS = bar;
    this.capS = el('p', 'hcc-note lab-cap', p.root);
  }

  buildFourier(slot, kind) {
    const S = this.app.S, app = this.app;
    const p = this['pf' + slot] = this.panel('Fourier');
    const abs = kind === 'fabs';
    el('span', 'lab-ptitle', p.head, abs ? '|δ̂(k)| of' : 'phase of δ̂(k) of');
    const s = sel(p.head, { options: FIELDS.map(([v, l]) => [v, l.replace(' density', '').replace(' 1+Dδ0', '')]), value: S.fo, onChange: (v) => {
      S.fo = v; app.hashChanged(); this['fsel' + (slot === 2 ? 3 : 2)].set(v);
      app.runPanel(this, 2); app.runPanel(this, 3);
    } });
    this['fsel' + slot] = s;
    const cmapKey = abs ? 'c3' : 'c4';
    sel(p.opts, { label: 'map', options: CMAPS, value: S[cmapKey], onChange: (v) => { S[cmapKey] = v; app.hashChanged(); this.updateFourier(slot); } });
    const { cv, bar } = this.canvasBlock(p.root);
    this['cvF' + slot] = cv; this['barF' + slot] = bar;
    this['capF' + slot] = el('p', 'hcc-note lab-cap', p.root);
    this['hover' + slot] = el('div', 'lab-hover hcc-note', p.root, ' ');
    cv.addEventListener('pointermove', (ev) => {
      const fv = this['fv' + slot]; if (!fv) return;
      const q = fv.pick(ev);
      this['hover' + slot].textContent = q ? `kx=${q.col - (fv.m >> 1)}, ky=${q.row - (fv.m >> 1)}: ${q.value.toPrecision(4)}` : ' ';
    });
  }

  markStale(b) { this.grid.querySelectorAll('.lab-panel').forEach((p) => p.classList.toggle('is-stale', b)); }
  fastUpdate() { return false; }
  redrawAll() { [0, 1, 2, 3].forEach((i) => this.app.runPanel(this, i)); }

  tasks(P) {
    return [
      { label: 'volume', heavy: true, fn: () => this.updateVolume(P) },
      { label: 'slice', heavy: false, fn: () => this.updateSlice(P) },
      { label: 'Fourier amplitude', heavy: true, fn: () => this.updateFourier(2, P) },
      { label: 'Fourier phase', heavy: false, fn: () => this.updateFourier(3, P) },
    ];
  }

  /** Used by app.runPanel for single-panel refreshes. */
  async update(i, P) {
    if (i === 0) return this.updateVolume(P);
    if (i === 1) return this.updateSlice(P);
    return this.updateFourier(i, P);
  }

  P() { return this.app.P; }

  async updateVolume(P = this.P()) {
    const S = this.app.S, eng = this.app.eng;
    if (!eng.sim) return;
    if (!this.vv) { this.vv = new VolumeView(this.cvV, { cmap: S.c1 }); this.vvInit = this.vv.init(); }
    await this.vvInit;
    const rho = eng.rho(S.v1, P);
    this.vv.setVolume(rho, eng.n);
    this.vv.setRange(undefined, undefined, { log: S.l1 });
    this.capV.textContent = `${FNAME[S.v1]}: ${S.vm === 'mip' ? 'maximum-intensity projection' : 'emission-absorption ray marching'} through the ${eng.n}³ box.`;
    this.drawVolume();
    this.pv.root.classList.remove('is-stale');
    this.updateSliceRange();
  }
  drawVolume() {
    const S = this.app.S;
    if (!this.vv || !this.vv.data) return;
    this.vv.setColormap(S.c1);
    this.vv.draw({ mode: S.vm, opacity: S.vo });
    const r = this.vv.getRange();
    renderColorbar(this.barV, S.c1, r.vmin, r.vmax, { label: 'ρ/ρ̄', log: r.log });
  }

  async updateSlice(P = this.P()) {
    const S = this.app.S, eng = this.app.eng;
    if (!eng.sim) return;
    if (!this.fvS) { this.fvS = new FieldView(this.cvS, { cmap: S.c2 }); this.fvSInit = this.fvS.init(); }
    await this.fvSInit;
    const n = eng.n;
    const idx = Math.max(0, Math.min(n - 1, Math.round(S.si * (n - 1))));
    const rho = eng.rho(S.s1, P);
    this.fvS.setField(slice3D(rho, n, S.sa, idx), n, n);
    this.fvS.setColormap(S.c2);
    let lo, hi;
    if (S.same && this.vv && this.vv.data && S.s1 === S.v1) { const r = this.vv.getRange(); lo = r.vmin; hi = r.vmax; }
    this.fvS.setRange(S.l2 ? lo : undefined, S.l2 ? hi : undefined, { log: S.l2 });
    this.fvS.draw();
    const r = this.fvS.getRange();
    renderColorbar(this.barS, S.c2, r.vmin, r.vmax, { label: 'ρ/ρ̄', log: r.log });
    const ax = ['x', 'y', 'z'][S.sa], hv = [['y', 'z'], ['x', 'z'], ['x', 'y']][S.sa];
    this.capS.textContent = `${FNAME[S.s1]} on the plane ${ax} = ${idx} of ${n} (${hv[0]} horizontal, ${hv[1]} vertical).`;
    this.ps.root.classList.remove('is-stale');
  }
  updateSliceRange() { if (this.app.S.same && this.fvS && this.fvS.data) this.updateSlice(); }

  async updateFourier(slot, P = this.P()) {
    const S = this.app.S, eng = this.app.eng;
    if (!eng.sim) return;
    const abs = slot === 2;
    const key = 'fv' + slot;
    if (!this[key]) { this[key] = new FieldView(this['cvF' + slot], { cmap: abs ? S.c3 : S.c4 }); this[key + 'Init'] = this[key].init(); }
    await this[key + 'Init'];
    const fv = this[key], n = eng.n;
    const m = eng.fmaps(S.fo, P);
    const arr = abs ? m.amp : m.phase;
    // fourier_* maps are [ikx*n + iky]; transpose so that kx is horizontal
    const t = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) t[j * n + i] = arr[i * n + j];
    fv.setField(t, n, n);
    const cmap = abs ? S.c3 : S.c4;
    fv.setColormap(cmap);
    if (abs) fv.setRange(-6, 0); else fv.setRange(-Math.PI, Math.PI);
    fv.draw();
    const r = fv.getRange();
    renderColorbar(this['barF' + slot], cmap, r.vmin, r.vmax, { label: abs ? 'log₁₀|δ̂|/max' : 'arg δ̂ [rad]' });
    const what = { cic: 'CIC', hc: 'Hopf–Cole', lin: 'linear' }[S.fo];
    this['capF' + slot].textContent = abs
      ? `log₁₀|δ̂(k)|/max of the ${what} density in the k_z = 0 plane (k_x horizontal, k_y vertical, k = 0 at the centre).`
      : `Phase of δ̂(k) of the ${what} density in the k_z = 0 plane. Mode coupling correlates the phases of generated modes with those of their parents.`;
    this['pf' + slot].root.classList.remove('is-stale');
  }

  renderBars() {
    const S = this.app.S;
    if (this.vv && this.vv.data) { const r = this.vv.getRange(); renderColorbar(this.barV, S.c1, r.vmin, r.vmax, { label: 'ρ/ρ̄', log: r.log }); }
    if (this.fvS && this.fvS.data) { const r = this.fvS.getRange(); renderColorbar(this.barS, S.c2, r.vmin, r.vmax, { label: 'ρ/ρ̄', log: r.log }); }
    for (const slot of [2, 3]) {
      const fv = this['fv' + slot];
      if (fv && fv.data) { const r = fv.getRange(); renderColorbar(this['barF' + slot], slot === 2 ? S.c3 : S.c4, r.vmin, r.vmax, { label: slot === 2 ? 'log₁₀|δ̂|/max' : 'arg δ̂ [rad]' }); }
    }
  }

  destroy() {
    this.ro.disconnect();
    for (const v of [this.vv, this.fvS, this.fv2, this.fv3]) { try { if (v) v.destroy(); } catch (e) { /* ignore */ } }
    this.root.remove();
  }
}
