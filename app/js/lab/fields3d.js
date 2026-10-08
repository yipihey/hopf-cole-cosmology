// fields3d.js - the "Fields" section in 3D: volume rendering, a slice and the k_z = 0 Fourier plane.

import { FieldView } from '../viz/fieldview.js';
import { VolumeView } from '../viz/volumeview.js';
import { renderColorbar, COLORMAPS } from '../viz/colormaps.js';
import { slider, checkbox } from '../viz/ui.js';
import { slice3D } from '../hcc.js';
import { el, sel } from './dom.js';

const CMAPS = Object.keys(COLORMAPS).map((c) => [c, c]);
const FIELDS = [['cic', 'CIC density'], ['sheet', 'Sheet density (tetrahedra)'], ['sheetp1', 'Sheet density (P1, vertex-interpolated)'], ['hc', 'Hopf–Cole density'], ['hcdual', 'Hopf–Cole dual sheet (mass-conserving)'], ['lin', 'Linear density 1+Dδ0']];
const DUAL_NOTE = ' The mass of a cell is the Lagrangian volume of its preimage under the Hopf–Cole inverse map (the hexahedron spanned by q at its eight corners, six Kuhn tetrahedra), so mass is conserved exactly and peaks are not under-resolved by finite differences; before shell crossing it should beat the finite-difference Hopf–Cole density in peaks and the forward sheet in voids.';
const FNAME = { cic: 'CIC density', sheet: 'tetrahedral sheet density (Kuhn simplices, point-sampled)', sheetp1: 'P1 sheet density (tetrahedra, linear vertex-interpolated shape, mass-conserving)', hc: 'Hopf–Cole density 1+δ', hcdual: 'Hopf–Cole dual sheet density (mass-conserving)', lin: 'linear density 1+Dδ0 (clipped at 10⁻³)' };

export class Fields3D {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-fields3d', host);
    this.volOn = !app.lite;          // lite: the WebGPU volume ray marcher is off by default (slice view); the user can turn it on
    this.volUser = false;
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
    this.volCb = checkbox(p.opts, { label: 'render volume', value: this.volOn, onChange: (v) => { this.volOn = v; this.volUser = true; app.runPanel(this, 0); } });
    this.volCb.el.title = 'Volume rendering (ray marching, or three central slices without WebGPU). Off by default in lite mode: use the slice panel.';
    this.phV = el('p', 'hcc-note lab-hint', p.root, 'Volume rendering is off (lite mode): the slice panel shows the field. Tick “render volume” to draw it anyway.');
    this.phV.hidden = true;
    const { cv, bar } = this.canvasBlock(p.root);
    this.cvV = cv; this.barV = bar;
    this.syncBackend();
    this.capV = el('p', 'hcc-note lab-cap', p.root);
    this.hintV = el('p', 'hcc-note lab-hint', p.root, 'Drag to rotate, wheel to zoom, double-click to reset (WebGPU). Without WebGPU: three central slices.');
  }

  buildSlice() {
    const S = this.app.S, app = this.app;
    const p = this.ps = this.panel('Slice');
    sel(p.head, { options: FIELDS, value: S.s1, onChange: (v) => { S.s1 = v; app.hashChanged(); app.runPanel(this, 1); } });
    this.axSel = sel(p.opts, { label: 'axis ⟂', options: [[0, 'x'], [1, 'y'], [2, 'z']], value: S.sa, onChange: (v) => { S.sa = Number(v); app.hashChanged(); this.updateSlice(); if (app.legendre) app.legendre.sliceChanged(); } });
    this.idxSl = slider(p.opts, { label: 'index', min: 0, max: Math.max(1, app.S.n - 1), step: 1, value: Math.round(S.si * (S.n - 1)), format: (v) => String(Math.round(v)),
      onInput: (v) => { S.si = v / (S.n - 1); app.hashChanged(); this.updateSlice(); if (app.legendre) app.legendre.sliceChanged(); } });
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
    const s = sel(p.head, { options: FIELDS.map(([v, l]) => [v, l.replace(' density', '').replace(' 1+Dδ0', '').replace(' (mass-conserving)', '')]), value: S.fo, onChange: (v) => {
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

  /** Called once the WebGPU probe has finished: the Canvas2D fallback shows three slices side by side. */
  syncBackend() { this.cvV.classList.toggle('hcc-wide', !this.app.gpu); }
  /** The performance preset changed: re-apply the volume default unless the user chose. */
  syncPerf() { if (!this.volUser) { this.volOn = !this.app.lite; this.volCb.set(this.volOn); } }

  /** Reflect S.sa / S.si changed elsewhere (Legendre lab). */
  syncSlice() { const S = this.app.S; this.axSel.set(S.sa); this.idxSl.set(Math.round(S.si * (S.n - 1))); }
  markStale(b) { this.grid.querySelectorAll('.lab-panel').forEach((p) => p.classList.toggle('is-stale', b)); }
  fastUpdate() { return false; }
  redrawAll() { [0, 1, 2, 3].forEach((i) => this.app.runPanel(this, i)); }

  /** Panels whose content comes from the Hopf-Cole solution (for the fast nu path). */
  dependsOnHc() {
    const S = this.app.S, h = (w) => w === 'hc' || w === 'hcdual';
    return { vol: h(S.v1), slice: h(S.s1), fourier: h(S.fo) }; }

  tasks(P, hcOnly = false) {
    // With the GPU path warm every panel is cheap (tens of ms): no per-task repaint waits, volume first.
    const S = this.app.S, e = this.app.eng, warm = e.gpuActive(P) && !!e.g && !!e.ref;
    const dep = this.dependsOnHc();
    const t = [
      { label: 'volume', heavy: !warm, fn: () => this.updateVolume(P), hc: dep.vol },
      { label: 'slice', heavy: (S.s1 === 'sheet' || S.s1 === 'sheetp1') && !warm, fn: () => this.updateSlice(P), hc: dep.slice },
      { label: 'Fourier amplitude', heavy: !warm, fn: () => this.updateFourier(2, P), hc: dep.fourier },
      { label: 'Fourier phase', heavy: false, fn: () => this.updateFourier(3, P), hc: dep.fourier },
    ];
    return hcOnly ? t.filter((q) => q.hc) : t;
  }

  /** Mark only the Hopf-Cole panels stale (nu changed). */
  markStaleHc(b) {
    const d = this.dependsOnHc();
    this.pv.root.classList.toggle('is-stale', b && d.vol);
    this.ps.root.classList.toggle('is-stale', b && d.slice);
    this.pf2.root.classList.toggle('is-stale', b && d.fourier);
    this.pf3.root.classList.toggle('is-stale', b && d.fourier);
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
    const wb = this.cvV.closest('.hcc-with-bar');
    wb.hidden = !this.volOn; this.phV.hidden = this.volOn; this.hintV.hidden = !this.volOn;
    if (!this.volOn) {
      this.capV.textContent = '';
      this.pv.root.classList.remove('is-stale');
      return;
    }
    if (!this.vv) { this.vv = new VolumeView(this.cvV, { cmap: S.c1 }); this.vvInit = this.vv.init(); }
    await this.vvInit;
    await eng.need('field', S.v1, P);
    const rho = eng.rho(S.v1, P);
    this.vv.setVolume(rho, eng.n);
    this.vv.setRange(undefined, undefined, { log: S.l1 });
    const gpuCic = S.v1 === 'cic' && eng.lastPath === 'GPU' && eng.gpuActive(P);
    this.capV.textContent = `${FNAME[S.v1]}: ${S.vm === 'mip' ? 'maximum-intensity projection' : 'emission-absorption ray marching'} through the ${eng.n}³ box.`
      + (S.v1 === 'sheetp1' ? ' The density inside each tetrahedron varies linearly between the vertex values 1/|J| (barycentric interpolation, rescaled so every simplex still deposits exactly its mass)' + (eng.lastPath === 'GPU' && eng.gpuActive(P) ? '; GPU: the vertex densities come from a finite-difference Jacobian of the displacement.' : '; WASM (GPU compute off): about 1 s at 64³.') : '')
      + (S.v1 === 'sheet' ? (eng.lastPath === 'GPU' && eng.gpuActive(P) ? ' GPU: one thread per Lagrangian cell, six Kuhn tetrahedra, watertight point-in-tetrahedron tests, 18-bit fixed-point atomics.' : ` WASM (GPU compute off): about 1 s at 64³, 4 s at 96³${eng.n >= 128 ? '; at 128³ this takes tens of seconds, prefer CIC or Hopf–Cole' : ''}.`) : '')
      + (S.v1 === 'hcdual' ? DUAL_NOTE : '')
      + (gpuCic ? ' GPU CIC deposits 18-bit fixed-point weights with integer atomics (mass conserved exactly); a cell would overflow at ρ/ρ̄ ≥ 16384.' : '');
    this.drawVolume();
    this.pv.root.classList.remove('is-stale');
    this.updateSliceRange();
  }
  drawVolume() {
    const S = this.app.S;
    if (!this.volOn || !this.vv || !this.vv.data) return;
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
    await eng.need('field', S.s1, P);
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
    this.capS.textContent = `${FNAME[S.s1]} on the plane ${ax} = ${idx} of ${n} (${hv[0]} horizontal, ${hv[1]} vertical).` + (S.s1 === 'hcdual' ? DUAL_NOTE : '');
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
    await eng.need('analysis', S.fo, P);
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
    const what = { cic: 'CIC', sheet: 'sheet', sheetp1: 'P1 sheet', hc: 'Hopf–Cole', hcdual: 'Hopf–Cole dual sheet', lin: 'linear' }[S.fo];
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
