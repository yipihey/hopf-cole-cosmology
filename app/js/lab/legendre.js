// legendre.js - the "Legendre lab" section: the Legendre-transform (Hopf-Lax) inversion of the nLPT map
// compared with the phase-space sheet and with the Zel'dovich (Burgers) Hopf-Cole density.
//
// Fields are looked up through the Engine by name: 'sheet' (reference), 'hcl' (Legendre / nLPT Hopf-Cole;
// the transverse-corrected variant when S.hs = 'lptT') and 'hcz' (Zel'dovich Hopf-Cole).  In 2D the panels
// show the whole field, in 3D the slice selected here (kept in sync with the slice panel of the Fields section).

import { FieldView } from '../viz/fieldview.js';
import { renderColorbar } from '../viz/colormaps.js';
import { LinePlot, PALETTE } from '../viz/plot.js';
import { slider, readout } from '../viz/ui.js';
import { transpose2D, slice3D, percentiles } from '../hcc.js';
import { el, sel, fmtNum } from './dom.js';

const DENS_CMAP = 'magma', DIFF_CMAP = 'rdbu';

class Panel {
  constructor(lab, host, { id, title, cmap, diff }) {
    this.lab = lab; this.id = id; this.cmap = cmap; this.diff = !!diff;
    this.root = el('div', 'hcc-panel lab-panel', host);
    this.titleEl = el('div', 'lab-ph lab-ptitle', this.root, title);
    const wb = el('div', 'hcc-with-bar', this.root);
    this.stage = el('div', 'hcc-stage', wb);
    this.cv = el('canvas', 'lab-cv', this.stage);
    this.bar = el('canvas', 'hcc-colorbar', wb); this.bar.width = 84; this.bar.height = 8;
    this.hover = el('div', 'lab-hover hcc-note', this.root, ' ');
    this.cap = el('p', 'hcc-note lab-cap', this.root);
    this.fv = null;
    this.cv.addEventListener('pointermove', (ev) => {
      if (!this.fv || !this.fv.data) return;
      const p = this.fv.pick(ev);
      if (!p) { this.hover.textContent = ' '; return; }
      const n = this.fv.m;
      this.hover.textContent = `x=(${((p.col + 0.5) / n).toFixed(3)}, ${((p.row + 0.5) / n).toFixed(3)}): ${fmtNum(p.value, 4)}`;
    });
    this.cv.addEventListener('pointerleave', () => { this.hover.textContent = ' '; });
  }
  async ensure() {
    if (!this.fv) { this.fv = new FieldView(this.cv, { cmap: this.cmap, interpolate: false }); this.init = this.fv.init(); }
    await this.init;
    return this.fv;
  }
  /** arr: Float32Array in FieldView layout (row*n+col). range: [lo, hi] (log density range, or symmetric difference range). */
  async show(arr, n, range) {
    const fv = await this.ensure();
    fv.setField(arr, n, n);
    fv.setColormap(this.cmap);
    if (this.diff) fv.setRange(-range[1], range[1]);
    else fv.setRange(range[0], range[1], { log: true });
    fv.draw();
    this.redrawBar();
    this.root.classList.remove('is-stale');
  }
  redrawBar() {
    if (!this.fv || !this.fv.data) return;
    const r = this.fv.getRange();
    renderColorbar(this.bar, this.cmap, r.vmin, r.vmax, { label: this.diff ? 'Δ(1+δ)' : '1+δ', log: r.log });
  }
  destroy() { try { if (this.fv) this.fv.destroy(); } catch (e) { /* ignore */ } this.root.remove(); }
}

const ordName = (o) => `${o}LPT`;

export class LegendreLab {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-legendre', host);
    el('p', 'hcc-note lab-leg-intro', this.root,
      'Does the Hopf–Cole machinery reproduce the exact nLPT density? The Legendre transform inverts any gradient Lagrangian map x = q + ∇S(q): '
      + 'the inverse is the gradient of the Legendre transform of q²/2 + S(q), and the Eulerian density is det(I − ∇∇Φ). '
      + 'The sheet below is the multi-stream reference; the choice of source potential (Dynamics → HC source ϕ) sets the transverse treatment.');

    // ---- toolbar
    const bar = this.bar = el('div', 'hcc-controls lab-fbar', this.root);
    this.rmin = slider(bar, { label: 'ρ min', min: 0.01, max: 1, value: S.rmin, log: true, onInput: (v) => { S.rmin = v; this.rangeChanged(); } });
    this.rmax = slider(bar, { label: 'ρ max', min: 3, max: 1000, value: S.rmax, log: true, onInput: (v) => { S.rmax = v; this.rangeChanged(); } });
    this.sliceBar = el('span', 'lab-leg-slice', bar);
    this.axSel = sel(this.sliceBar, { label: 'slice axis ⟂', options: [[0, 'x'], [1, 'y'], [2, 'z']], value: S.sa, onChange: (v) => { S.sa = Number(v); this.sliceMoved(); } });
    this.idxSl = slider(this.sliceBar, { label: 'index', min: 0, max: 1, step: 0.002, value: S.si,
      format: (v) => String(Math.round(v * Math.max(1, this.app.S.n - 1))), onInput: (v) => { S.si = v; this.sliceMoved(); } });
    this.warn3 = el('p', 'hcc-note lab-leg-status', this.root);
    this.warn3.hidden = true;
    this.warnBig = el('p', 'lab-warn-note', this.root, 'Without the GPU compute path the tetrahedral sheet runs in WASM and is slow above 64³ (tens of seconds at 128³; prefer 32³–64³ or switch GPU compute on).');
    this.warnBig.hidden = true;
    this.status = el('p', 'hcc-note lab-leg-status', this.root);

    // ---- density panels
    const g1 = el('div', 'hcc-grid2 lab-grid lab-leg-grid', this.root);
    this.pSheet = new Panel(this, g1, { id: 'sheet', title: 'Sheet density', cmap: DENS_CMAP });
    this.pLeg = new Panel(this, g1, { id: 'leg', title: 'Legendre Hopf–Cole density', cmap: DENS_CMAP });
    this.pZel = new Panel(this, g1, { id: 'zel', title: 'Zel’dovich Hopf–Cole density', cmap: DENS_CMAP });
    // ---- difference panels
    const g2 = el('div', 'hcc-grid2 lab-grid lab-leg-grid lab-leg-grid2', this.root);
    this.pdLeg = new Panel(this, g2, { id: 'dleg', title: 'Legendre − sheet', cmap: DIFF_CMAP, diff: true });
    this.pdZel = new Panel(this, g2, { id: 'dzel', title: 'Zel’dovich HC − sheet', cmap: DIFF_CMAP, diff: true });
    this.panels = [this.pSheet, this.pLeg, this.pZel, this.pdLeg, this.pdZel];

    // ---- readouts
    const ro = el('div', 'hcc-controls lab-readouts', this.root);
    this.ro = {
      leg: readout(ro, { label: 'rms(Legendre − sheet) / rms(sheet − 1)' }),
      zel: readout(ro, { label: 'rms(ZelHC − sheet) / rms(sheet − 1)' }),
      psit: readout(ro, { label: 'Ψ_T / Ψ_L (rms)' }),
      dd: readout(ro, { label: 'D / D_sc(order)' }),
    };
    this.ro.leg.el.title = 'Root-mean-square difference between the Legendre-transform Hopf–Cole density and the sheet density, relative to the rms density contrast of the sheet. Over the full box (2D) or volume (3D).';
    this.ro.zel.el.title = 'The same for the Zel’dovich Hopf–Cole density (Burgers solution): what is missing relative to the exact nLPT map is the higher-order displacement.';
    this.ro.psit.el.title = 'rms of the transverse (curl) part of the displacement over the rms of the longitudinal part; zero through 2LPT where Ψ is exactly a gradient.';
    this.ro.dd.el.title = 'Growth factor in units of the first shell-crossing time of the chosen LPT order; above 1 the sheet has multiple streams.';

    // ---- spectra
    const grid = el('div', 'lab-plots-grid', this.root);
    const pp = this.pPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', pp, 'Power spectra: sheet, Legendre, Zel’dovich HC');
    this.pkPlot = new LinePlot(el('div', 'lab-plot', pp), { width: 500, height: 340 });
    this.pkCap = el('p', 'hcc-note lab-cap', pp);
    const rp = this.rPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', rp, 'Ratio to the sheet spectrum');
    this.ratioPlot = new LinePlot(el('div', 'lab-plot', rp), { width: 500, height: 340 });
    el('p', 'hcc-note lab-cap', rp, 'P_Legendre/P_sheet and P_ZelHC/P_sheet on a linear axis (0.5–1.5). Before shell crossing the Legendre ratio stays within a few per cent of one up to the smoothing scale 1/R; the Zel’dovich ratio departs from one where the 2LPT and higher displacement terms matter. Beyond about one over the sheet’s rasterization scale the sheet itself is noisy.');

    this.lastP = null;
    this.syncMode();
  }

  // ---------------------------------------------------------------- interaction

  syncMode() {
    const d3 = this.app.S.mode === 3;
    this.sliceBar.hidden = !d3;
    this.axSel.set(this.app.S.sa); this.idxSl.set(this.app.S.si);
    this.warn3.hidden = !d3;
    this.warn3.textContent = d3 ? 'Reference: the tetrahedral sheet density (six Kuhn simplices per Lagrangian cell, point-sampled). With GPU compute on, the nLPT build, the sheet, the Legendre inversion (including the transverse correction) and the spectra all run in WebGPU compute shaders (milliseconds at 64³–128³); otherwise WASM (about 1 s at 64³). Slices share the axis and index of the slice panel in the Fields section.' : '';
  }
  rangeChanged() {
    this.app.hashChanged();
    const f = this.app.fields;
    if (f && f.syncRange) f.syncRange();
    this.redraw();
  }
  sliceMoved() {
    const S = this.app.S;
    this.app.hashChanged();
    const f = this.app.fields;
    if (f && f.syncSlice) f.syncSlice();
    if (f && f.updateSlice && this.app.visible('f') && this.app.eng.sim) f.updateSlice();
    this.redraw();
  }
  /** The slice controls of the Fields section changed. */
  sliceChanged() {
    const S = this.app.S;
    this.axSel.set(S.sa); this.idxSl.set(S.si);
    this.redraw();
  }
  /** Redraw from cache (display-only change). */
  redraw() {
    if (!this.app.visible('l') || !this.app.eng.sim || !this.lastP || this.app.eng.dim !== this.app.S.mode) return;
    this.draw(this.lastP).catch((err) => this.app.reportError('Legendre lab', err));
  }
  markStale(b) { for (const p of this.panels) p.root.classList.toggle('is-stale', b); this.pPanel.classList.toggle('is-stale', b); this.rPanel.classList.toggle('is-stale', b); }

  // ---------------------------------------------------------------- computation

  tasks(P) {
    const e = this.app.eng, d3 = P.mode === 3;
    const t = [];
    const gpu3 = d3 && e.gpuActive(P);
    this.warnBig.hidden = !(d3 && !gpu3 && P.n > 64);
    t.push({ label: d3 ? (gpu3 ? 'Legendre lab: sheet (tetrahedra, GPU)' : 'Legendre lab: sheet (tetrahedra, CPU)') : 'Legendre lab: sheet', heavy: d3 && !gpu3,
      fn: async () => { await e.need('field', 'sheet', P); e.delta('sheet', P); } });
    t.push({ label: 'Legendre lab: Hopf–Cole (Legendre, Zel’dovich)', fn: async () => {
      await e.need('field', 'hcl', P); await e.need('field', 'hcz', P);
      e.delta('hcl', P); e.delta('hcz', P); e.psiT(P);
    } });
    t.push({ label: 'Legendre lab: spectra', fn: async () => {
      e.pk('lin', P);
      await e.need('analysis', 'sheet', P); e.pk('sheet', P, false);
      await e.need('analysis', 'hcl', P); e.pk('hcl', P, false);
      await e.need('analysis', 'hcz', P); e.pk('hcz', P, false);
    } });
    t.push({ label: 'Legendre lab: draw', fn: () => this.draw(P) });
    return t;
  }

  /** Extract the displayed 2D array (FieldView layout) from a full field. */
  view(arr, P) {
    const S = this.app.S, n = P.n;
    if (P.mode === 2) return transpose2D(arr, n);
    const idx = Math.max(0, Math.min(n - 1, Math.round(S.si * (n - 1))));
    return slice3D(arr, n, S.sa, idx);
  }

  async draw(P) {
    const e = this.app.eng, S = this.app.S;
    if (!e.sim || e.dim !== P.mode) return;
    this.lastP = P;
    const n = P.n, order = P.order;
    const dS = e.delta('sheet', P), dL = e.delta('hcl', P), dZ = e.delta('hcz', P);
    const ref = Math.max(rmsOf(dS), 1e-12);
    const diffL = subtract(dL, dS), diffZ = subtract(dZ, dS);
    const rl = rmsOf(diffL) / ref, rz = rmsOf(diffZ) / ref;
    const range = [S.rmin, S.rmax];
    const vL = this.view(diffL, P), vZ = this.view(diffZ, P);
    // shared symmetric range of the two difference maps
    const q = (a) => { const abs = new Float32Array(a.length); for (let i = 0; i < a.length; i++) abs[i] = Math.abs(a[i]); return percentiles(abs, 0, 99.5)[1]; };
    const dr = Math.max(q(vL), q(vZ), 1e-3);

    const lpt = P.hs === 'lptT' ? `${ordName(order)}${order > 2 ? ' + transverse' : ''}` : ordName(order);
    const where = P.mode === 3 ? ` on the plane ${['x', 'y', 'z'][S.sa]} = ${Math.max(0, Math.min(n - 1, Math.round(S.si * (n - 1))))} of ${n}` : '';
    this.pSheet.titleEl.textContent = `Sheet density (${ordName(order)})`;
    this.pLeg.titleEl.textContent = `Legendre Hopf–Cole (${lpt})`;
    this.pZel.titleEl.textContent = 'Zel’dovich Hopf–Cole (1LPT)';
    this.pdLeg.titleEl.textContent = 'Legendre − sheet';
    this.pdZel.titleEl.textContent = 'Zel’dovich HC − sheet';
    const nuTxt = `ν = ${P.nu.toExponential(1)}`;
    this.pSheet.cap.textContent = `Multi-stream reference: Σ 1/|J| over all streams of the ${ordName(order)} map x(q,D)${P.mode === 3 ? ' (Kuhn tetrahedra)' : ' (2×2 supersampled triangles)'}${where}. It carries rasterization noise at the cell scale that shrinks with N, and after shell crossing it adds the streams.`;
    this.pLeg.cap.textContent = `1+δ = det(I − ∇∇Φ), Φ the Legendre transform (Hopf–Lax minimum, heat-kernel smoothed: ${nuTxt}) of q²/2 + S(q) for the ${lpt} potential S; particle-free, on the Eulerian grid. Before shell crossing this is the exact ${ordName(order)} density; afterwards it keeps one stream per point (adhesion), so walls replace the multi-stream regions.`;
    this.pZel.cap.textContent = `The same solver with the Zel’dovich potential S = −Dϕ: the Burgers/adhesion solution (${nuTxt}). It is exact for the Zel’dovich map and misses the 2LPT and higher displacement, so filaments sit slightly off.`;
    this.pdLeg.cap.textContent = `Δ(1+δ) = Legendre − sheet, symmetric colour scale ±${fmtNum(dr, 2)}. Before shell crossing the residual is the sheet’s rasterization noise (shrinks with N) plus ν-smoothing${order > 2 ? ' and the neglected or approximately corrected transverse displacement' : ''}; after shell crossing the sheet sums streams, so it exceeds the Legendre density inside the folds.`;
    this.pdZel.cap.textContent = `Δ(1+δ) = Zel’dovich HC − sheet on the same scale. Here the residual is dominated by what 1LPT lacks: the 2LPT and higher displacement shifts the filaments and changes their amplitude; compare the amplitude with the Legendre map.`;

    await this.pSheet.show(this.view(rho(e, 'sheet', P), P), n, range);
    await this.pLeg.show(this.view(rho(e, 'hcl', P), P), n, range);
    await this.pZel.show(this.view(rho(e, 'hcz', P), P), n, range);
    await this.pdLeg.show(vL, n, [0, dr]);
    await this.pdZel.show(vZ, n, [0, dr]);

    this.ro.leg.set(fmtNum(rl, 3));
    this.ro.zel.set(fmtNum(rz, 3));
    const ratio = e.psiT(P);
    this.ro.psit.set(order <= 2 ? '0 (Ψ is a pure gradient)' : fmtNum(ratio, 2));
    this.updateReadouts(P);
    this.drawSpectra(P);
    this.markStale(false);
  }

  /** D / D_sc and the explanatory status line; cheap, called after every run from the app. */
  updateReadouts(P) {
    const e = this.app.eng;
    if (!e.sim || e.dim !== P.mode) return;
    const dsc = e.peek(['dsc', P.order]);
    const known = dsc !== undefined && Number.isFinite(dsc) && dsc > 0;
    const r = known ? P.D / dsc : (dsc === Infinity ? 0 : NaN);
    this.ro.dd.set(Number.isFinite(r) ? fmtNum(r, 3) : '–');
    this.ro.dd.el.classList.toggle('lab-bad', r > 1);
    this.status.className = 'hcc-note lab-leg-status' + (r > 1 ? ' lab-bad' : '');
    this.status.textContent = !Number.isFinite(r) ? ''
      : r > 1 ? `D > D_sc(${P.order}): shell crossing has occurred. The sheet now has multi-stream regions (it sums the streams); the Legendre map keeps a single stream per point (adhesion), so large differences are expected inside the folds.`
        : `D < D_sc(${P.order}) (D/D_sc = ${fmtNum(r, 2)}): the map is one-to-one, the Legendre density is the exact ${ordName(P.order)} density and the Legendre − sheet residual is the sheet’s rasterization noise.`;
  }

  drawSpectra(P) {
    const e = this.app.eng, dim = e.dim;
    const kmax = Math.sqrt(dim) * e.knyq, kmin = 0.8 * e.kf;
    const sh = e.pk('sheet', P, false), le = e.pk('hcl', P, false), ze = e.pk('hcz', P, false), li = e.plinFine(P);
    const lpt = P.hs === 'lptT' && P.order > 2 ? 'Legendre (nLPT + transverse)' : 'Legendre (nLPT)';
    const series = [
      { x: li.k, y: li.p, label: 'linear D²P₀', color: '#8a8f98', width: 1.6, dash: '5 3' },
      { x: sh.k, y: sh.p, label: dim === 3 ? 'sheet (tetrahedra)' : 'sheet', color: PALETTE[0], points: true, width: 3, opacity: 0.55, radius: 3 },
      { x: le.k, y: le.p, label: lpt, color: PALETTE[2], points: true, width: 1.2, radius: 2.2 },
      { x: ze.k, y: ze.p, label: 'Zel’dovich HC', color: PALETTE[1], points: true, width: 1.2, radius: 2.2 },
    ];
    let top = 0;
    for (const s of series) for (let i = 0; i < s.y.length; i++) { const v = s.y[i]; if (v > top && Number.isFinite(v)) top = v; }
    const markers = [{ x: e.knyq, label: 'k_Nyq', color: '#888' }];
    if (P.R > 0) markers.push({ x: 1 / P.R, label: '1/R', color: '#aa66cc' });
    this.pkPlot.setAxes({ xlog: true, ylog: true, xlabel: 'k  [rad / L]', ylabel: `P(k)  [L^${dim}]`, xlim: [kmin, kmax], ylim: top > 0 ? [top * 1e-10, top * 2] : undefined });
    this.pkPlot.setSeries(series); this.pkPlot.setMarkers(markers); this.pkPlot.draw();
    this.pkCap.textContent = 'Measured binned spectra of the three densities (dashed vertical lines: k_Nyq and 1/R). The Hopf–Cole spectra follow the sheet at low k; at high k the sheet’s own rasterization floor and the ν-smoothing of the Hopf–Cole fields differ.';

    const ratio = (num) => {
      const x = [], y = [], m = Math.min(num.k.length, sh.k.length);
      for (let i = 0; i < m; i++) if (sh.p[i] > 0 && Number.isFinite(num.p[i])) { x.push(sh.k[i]); y.push(num.p[i] / sh.p[i]); }
      return { x, y };
    };
    const rl = ratio(le), rz = ratio(ze);
    this.ratioPlot.setAxes({ xlog: true, ylog: false, xlabel: 'k  [rad / L]', ylabel: 'P / P_sheet', xlim: [kmin, kmax], ylim: [0.5, 1.5] });
    this.ratioPlot.setSeries([
      { x: [kmin, kmax], y: [1, 1], label: '1', color: '#8a8f98', dash: '5 3', width: 1.2 },
      { ...rl, label: 'P_Legendre / P_sheet', color: PALETTE[2], points: true, width: 1.6, radius: 2.6 },
      { ...rz, label: 'P_ZelHC / P_sheet', color: PALETTE[1], points: true, width: 1.6, radius: 2.6 },
    ]);
    this.ratioPlot.setMarkers(markers);
    this.ratioPlot.draw();
  }

  resizeBars() { for (const p of this.panels) p.redrawBar(); }
  destroy() { for (const p of this.panels) p.destroy(); this.root.remove(); }
}

function rho(e, which, P) { return e.rho(which, P); }
function rmsOf(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); }
function subtract(a, b) { const o = new Float32Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] - b[i]; return o; }
