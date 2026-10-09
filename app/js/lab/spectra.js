// spectra.js - the "Spectra" section: P(k), r(k), EFT counterterm fit and readouts.

import { LinePlot, PALETTE } from '../viz/plot.js';
import { slider, checkbox, readout } from '../viz/ui.js';
import { fitCounterterm } from '../hcc.js';
import { el, sel, fmtNum, docLink } from './dom.js';
import { BOOK } from './catalog.js';

export const DUAL_COLOR = '#6a3d9a';
export const P1_COLOR = '#a0522d';
export const EXACT0_COLOR = '#9a9a00';
export const EXACT1_COLOR = '#0aa5b5';
export const DIRECT0_COLOR = '#d6336c';
export const DIRECT1_COLOR = '#2b8a3e';
export const NUFFT_COLOR = '#e8590c';

/** Series of SERIES that are fields (the phase sums exist only for these; the analytic curves and the direct sheet spectra have no phases). */
const PHASE_FIELDS = ['lin', 'sheet', 'sheetp1', 'sheetx', 'sheetxp1', 'nufft', 'cic', 'hc', 'hcdual'];
const PH_NBINS = 36;

const SERIES = [
  ['lin', 'linear theory'],
  ['sheet', 'sheet (measured)'],
  ['sheetp1', 'sheet P1 (vertex-interpolated)'],
  ['sheetx', 'sheet exact P0 (clipped)'],
  ['sheetxp1', 'sheet exact P1 (clipped)'],
  ['dir0', 'direct sheet spectrum (P0)'],
  ['dir1', 'direct sheet spectrum (P1)'],
  ['nufft', 'NUFFT density (refined map)'],
  ['cic', 'CIC (deconv.)'],
  ['hc', 'Hopf–Cole (selected source)'],
  ['hcdual', 'Hopf–Cole dual sheet'],
  ['spt', '1-loop SPT'],
  ['za', '1-loop Zel’dovich'],
  ['eft', 'EFT fit'],
  ['p22', 'P22'],
  ['p13', '|P13|'],
];

export class Spectra {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-spectra', host);
    const intro = el('p', 'hcc-note lab-leg-intro', this.root,
      'Power spectra of the selected density estimators against linear theory and the one-loop predictions, their cross-correlation r(k) with the linear field, and the distribution of Fourier phase sums.');
    docLink(intro, BOOK.pk);
    const grid = el('div', 'lab-plots-grid', this.root);

    // ---- P(k) panel
    const pp = this.pPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', pp, 'Power spectrum P(k)');
    const cbs = el('div', 'hcc-controls lab-checks', pp);
    this.checks = {};
    for (const [id, label] of SERIES) {
      this.checks[id] = checkbox(cbs, { label, value: S.ser.includes(id), onChange: (v) => {
        const set = new Set(S.ser); if (v) set.add(id); else set.delete(id);
        S.ser = SERIES.map((q) => q[0]).filter((q) => set.has(q));
        app.hashChanged(); app.runSpectra();
      } });
    }
    this.kmaxSl = slider(cbs, { label: 'EFT k_max', min: 8, max: 2000, value: S.km, log: true, format: (v) => v.toFixed(0),
      onInput: (v) => { S.km = v; app.hashChanged(); this.drawIfReady(); } });
    this.csOut = readout(cbs, { label: 'c_s²' });
    // direct (deposit-free) spectra and NUFFT density: modes per |k| bin and refinement of the Fourier-interpolated map
    this.dwCb = checkbox(cbs, { label: 'deconvolve cell window (exact deposits, dual sheet)', value: S.dw, onChange: (v) => { S.dw = !!v; app.hashChanged(); app.runSpectra(); } });
    this.dwCb.el.title = 'The exact sheet deposits and the Hopf–Cole dual sheet are cell averages, so their spectra carry the top-hat window prod_a sinc²(k_a dx/2); on: divide it out, off: show the raw cell-averaged spectra.';
    this.dmSl = slider(cbs, { label: 'modes / bin', min: 32, max: 512, value: S.dm, log: true, format: (v) => String(Math.round(v)),
      onInput: (v) => { S.dm = Math.round(v); app.hashChanged(); clearTimeout(this.dmTimer); this.dmTimer = setTimeout(() => app.modesChanged(), 250); } });
    this.dmSl.el.title = 'Lattice modes sampled per |k| bin for the direct sheet spectra (shells with fewer modes are enumerated completely). More modes: less scatter, proportionally more GPU time.';
    this.rfSel = sel(cbs, { label: 'NUFFT refine', options: [[1, '1 (n)'], [2, '2 (2n)'], [4, '4 (4n)']], value: S.rf,
      title: 'Spectral refinement of the Lagrangian grid for the NUFFT density: the displacement is Fourier-interpolated to refine x n points per axis before the trapezoidal rule. 3D: up to 64³ with refine ≤ 2.',
      onChange: (v) => { S.rf = Number(v); app.hashChanged(); app.refineChanged(); } });
    this.pkHost = el('div', 'lab-plot', pp);
    this.pkPlot = new LinePlot(this.pkHost, { width: 500, height: 380 });
    this.dirNote = el('p', 'lab-warn-note', pp); this.dirNote.hidden = true;
    this.dirInfo = el('p', 'hcc-note lab-cap lab-dirinfo', pp); this.dirInfo.hidden = true;
    this.dirCap = el('p', 'hcc-note lab-cap', pp, 'Direct sheet spectrum (P0 = constant, P1 = linear density per simplex): exact Fourier transform of the piecewise-linear sheet, sampled on lattice modes; no grid, no window, no aliasing — the k⁻³ facet tail is real.');
    this.nufftCap = el('p', 'hcc-note lab-cap', pp, 'NUFFT density: Lagrangian integral ∫d^dq e^{−ik·x(q)} of the Fourier-refined map by the periodic trapezoidal rule (type-1 NUFFT, Gaussian kernel, 2× oversampling); spectrally accurate for a band-limited displacement, counts all streams, no facets.');
    this.dirCap.hidden = true; this.nufftCap.hidden = true;
    const pkCap = el('p', 'hcc-note lab-cap', pp, 'Dashed vertical lines mark k_Nyq and the smoothing scale 1/R. The P1 sheet interpolates the vertex densities 1/|J| linearly inside every simplex (mass-conserving, second-order accurate), so it carries less high-k rasterization noise than the plain sheet. The exact sheet deposits clip every simplex against the cells it overlaps and integrate its (constant or vertex-interpolated) density over each piece: no sampling noise, mass conserved to roundoff, so their raw spectra sit below the point-sampled ones at high k, where only the facet structure of the simplices remains. The Hopf–Cole dual sheet is the mass-conserving density of the same inverse map (cell mass = Lagrangian volume of the cell’s preimage). Cell-averaged estimators (exact deposits, dual sheet) are divided by the top-hat window Π sinc²(k_aΔx/2); point-sampled sheets and the NUFFT/direct spectra carry no window. Linear theory is the smooth curve D²P0(k); the 1-loop curves are only available for Gaussian ICs. The EFT curve subtracts 2c_s²k²P_lin from the SPT 1-loop with c_s² fitted to the measured spectrum for k < k_max.');
    pkCap.appendChild(document.createTextNode(' ')); docLink(pkCap, BOOK.eft);

    // ---- r(k) panel
    const rp = this.rPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', rp, 'Cross-correlation with the linear field r(k)');
    this.rHost = el('div', 'lab-plot', rp);
    this.rPlot = new LinePlot(this.rHost, { width: 500, height: 380 });
    const rCap = el('p', 'hcc-note lab-cap', rp, 'r(k) = P_{f,lin} / √(P_f P_lin) measures how much of the evolved field’s phase information still matches linear theory. The dashed curve is the Zel’dovich propagator exp(−k²σ_Ψ²D²/2), with σ_Ψ² the per-axis displacement variance of the grid modes (Gaussian ICs only). It equals r(k) only while the evolved power is still close to linear, and is only approximate once streams cross.');
    rCap.appendChild(document.createTextNode(' ')); docLink(rCap, BOOK.phases);

    // ---- phase sums panel
    const hp = this.phPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', hp, 'Phase sums');
    const hc = el('div', 'hcc-controls lab-checks', hp);
    this.pswSel = sel(hc, { label: 'weight', options: [['n', 'none'], ['a', 'amplitude']], value: S.psw,
      title: 'none: every closed triangle counts equally (Hikage, Matsubara & Suto 2004). amplitude: weighted by |f(k1) f(k2) f(k1+k2)|, i.e. the distribution of the bispectrum phase.',
      onChange: (v) => { S.psw = v === 'a' ? 'a' : 'n'; app.hashChanged(); this.drawIfReady(); } });
    this.pskSel = sel(hc, { label: 'k_max', options: [8, 16, 32, 64].map((v) => [v, v + ' k_f']), value: S.psk,
      title: 'Largest side |k1|, |k2|, |k1+k2| of the triangles, in units of the fundamental k_f = 2π/L (clipped to the Nyquist mode of the grid).',
      onChange: (v) => { S.psk = Number(v); app.hashChanged(); this.drawIfReady(); } });
    this.phHost = el('div', 'lab-plot', hp);
    this.phPlot = new LinePlot(this.phHost, { width: 500, height: 380 });
    this.phVals = el('div', 'hcc-note lab-phvals', hp);
    const hCap = el('p', 'hcc-note lab-cap', hp, 'Distribution of the phase sum θ = φ(k₁) + φ(k₂) − φ(k₁+k₂) over random closed triangles with all sides below the chosen k_max, shown as 2π p(θ): a Gaussian field gives the dotted line at 1 (⟨cos θ⟩ = 0), while phase-locked structure (the harmonics of a wave, pancakes, nodes) piles up at θ = 0 and raises ⟨cos θ⟩.');
    hCap.appendChild(document.createTextNode(' ')); docLink(hCap, BOOK.phases);

    // ---- readouts
    const ro = this.roRow = el('div', 'hcc-controls lab-readouts', this.root);
    this.ro = {
      sigma: readout(ro, { label: 'σ0' }),
      dsc: readout(ro, { label: 'D_sc(order)' }),
      dsc1: readout(ro, { label: 'D_sc(1)' }),
      nu: readout(ro, { label: 'ν_eff' }),
      exp: readout(ro, { label: 'exponent range' }),
      modes: readout(ro, { label: 'modes' }),
    };
    this.cs2 = null;
    this.syncMode();
  }

  syncMode() {
    const d3 = this.app.S.mode === 3;
    this.checks.sheet.el.querySelector('.hcc-label').textContent = d3 ? 'sheet (tetrahedra)' : 'sheet (measured)';
    this.checks.sheetp1.el.querySelector('.hcc-label').textContent = d3 ? 'sheet P1 (tetrahedra)' : 'sheet P1 (vertex-interpolated)';
    this.checks.sheetx.el.querySelector('.hcc-label').textContent = d3 ? 'sheet exact P0 (tetrahedra)' : 'sheet exact P0 (clipped)';
    this.checks.sheetxp1.el.querySelector('.hcc-label').textContent = d3 ? 'sheet exact P1 (tetrahedra)' : 'sheet exact P1 (clipped)';
    this.checks.cic.el.querySelector('.hcc-label').textContent = d3 ? 'CIC (measured, deconv.)' : 'CIC (deconv.)';
    this.checks.nufft.el.querySelector('.hcc-label').textContent = d3 ? 'NUFFT density (refined map; ≤ 64³, refine ≤ 2)' : 'NUFFT density (refined map)';
    this.dwCb.set(this.app.S.dw);
    this.pswSel.set(this.app.S.psw); this.pskSel.set(this.app.S.psk);
    this.dmSl.set(this.app.S.dm); this.rfSel.set(this.app.S.rf);
    const ser = this.app.S.ser;
    for (const id of Object.keys(this.checks)) this.checks[id].set(ser.includes(id));
  }

  markStale(b) { this.pPanel.classList.toggle('is-stale', b); this.rPanel.classList.toggle('is-stale', b); this.phPanel.classList.toggle('is-stale', b); }

  wants() {
    const w = new Set(this.app.S.ser);
    return w;
  }

  /** Tasks that fill the caches and finally draw both plots. */
  tasks(P) {
    const e = this.app.eng, w = this.wants(), dw = this.app.S.dw;
    const dim = P.mode, gaussian = P.ic === 'g';      // engine state may not be configured yet when tasks are built
    const prim = dim === 2 ? 'sheet' : 'cic';
    const t = [];
    t.push({ label: 'P(k) linear', fn: () => { e.pk('lin', P); e.plinFine(P); } });
    // 3D GPU path: `need` runs the GPU FFT once per field and seeds P(k), r(k) and the Fourier maps
    if (w.has('sheet')) t.push({ label: dim === 3 ? 'P(k) sheet (tetrahedra)' : 'P(k) sheet', fn: async () => { await e.need('analysis', 'sheet', P); e.pk('sheet', P, false); } });
    if (w.has('sheetp1')) t.push({ label: dim === 3 ? 'P(k) sheet P1 (tetrahedra)' : 'P(k) sheet P1', fn: async () => { await e.need('analysis', 'sheetp1', P); e.pk('sheetp1', P, false); } });
    if (w.has('sheetx')) t.push({ label: 'P(k) sheet exact P0', heavy: dim === 3 && !e.gpuActive(P), fn: async () => { await e.need('analysis', 'sheetx', P); e.pk('sheetx', P, dw); } });
    if (w.has('sheetxp1')) t.push({ label: 'P(k) sheet exact P1', heavy: dim === 3 && !e.gpuActive(P), fn: async () => { await e.need('analysis', 'sheetxp1', P); e.pk('sheetxp1', P, dw); } });
    for (const [id, p1] of [['dir0', false], ['dir1', true]]) {
      if (!w.has(id)) continue;
      t.push({ label: `direct sheet spectrum (${p1 ? 'P1' : 'P0'}${e.directGpuPossible(P) ? ', GPU' : ''})`, heavy: !e.directGpuPossible(P),
        fn: async () => { await e.needDirect(p1, P); e.directPk(p1, P); } });
    }
    if (w.has('nufft')) t.push({ label: 'P(k) NUFFT density', heavy: true, fn: async () => {
      if (!e.nufftInfo(P).ok) return;
      await e.need('analysis', 'nufft', P); e.pk('nufft', P, false);
    } });
    if (w.has('cic')) t.push({ label: 'P(k) CIC', fn: async () => { await e.need('analysis', 'cic', P); e.pk('cic', P, true); } });
    if (w.has('hc')) t.push({ label: 'P(k) Hopf–Cole', fn: async () => { await e.need('analysis', 'hc', P); e.pk('hc', P); } });
    if (w.has('hcdual')) t.push({ label: 'P(k) Hopf–Cole dual sheet', fn: async () => { await e.need('analysis', 'hcdual', P); e.pk('hcdual', P, dw); } });
    t.push({ label: 'r(k)', fn: async () => {
      await e.need('analysis', prim, P); await e.need('analysis', 'hc', P); e.rk(prim, P); e.rk('hc', P); e.sigmaV2();
      if (w.has('hcdual')) { await e.need('analysis', 'hcdual', P); e.rk('hcdual', P); }
      if (w.has('sheetp1')) { await e.need('analysis', 'sheetp1', P); e.rk('sheetp1', P); }
      for (const m of ['sheetx', 'sheetxp1']) if (w.has(m)) { await e.need('analysis', m, P); e.rk(m, P); }
      if (w.has('nufft') && e.nufftInfo(P).ok) { await e.need('analysis', 'nufft', P); e.rk('nufft', P); }
    } });
    const loops = gaussian && (w.has('spt') || w.has('eft') || w.has('p22') || w.has('p13') || w.has('za'));
    // lite: show the measured spectra first, the 1-loop curves (0.3 s in WASM) follow in a second drawing; only when this section is visible
    if (this.app.lite && loops && this.app.visible('s') && !e.peek(['loop1', 0])) t.push({ label: 'plots (1-loop follows)', fn: () => this.draw(P, true) });
    if (gaussian && (w.has('spt') || w.has('eft') || w.has('p22') || w.has('p13'))) {
      t.push({ label: 'SPT 1-loop', heavy: true, fn: () => e.loop(P, 0) });
    }
    if (gaussian && w.has('za')) t.push({ label: 'Zel’dovich 1-loop', heavy: true, fn: () => e.loop(P, 1) });
    if (PHASE_FIELDS.some((id) => w.has(id))) t.push({ label: 'phase sums', fn: () => this.phaseSums(P) });
    t.push({ label: 'plots', fn: () => this.draw(P) });
    return t;
  }

  drawIfReady() { if (this.app.eng.sim && this.lastP) this.draw(this.lastP); }

  draw(P, noLoop = false) {
    const e = this.app.eng, S = this.app.S, w = this.wants(), dw = S.dw;
    this.lastP = P;
    const dim = e.dim, prim = dim === 2 ? 'sheet' : 'cic';
    const kmaxPlot = Math.sqrt(dim) * e.knyq, kminPlot = 0.8 * e.kf;
    const series = [];
    const hcTag = P.hs === 'zel' || !P.hs ? '' : P.hs === 'lptT' && P.order > 2 ? ` (Legendre ${P.order}LPT+T)` : ` (Legendre ${P.order}LPT)`;
    const primLabel = dim === 2 ? 'sheet' : 'CIC (deconv.)';

    const lin = e.plinFine(P);
    if (w.has('lin')) series.push({ x: lin.k, y: lin.p, label: 'linear theory D²P₀', color: '#8a8f98', width: 2.2 });
    if (w.has('sheet')) {
      const s = e.pk('sheet', P, false);
      series.push(dim === 2 ? { x: s.k, y: s.p, label: 'sheet', color: PALETTE[0], points: true, width: 3, opacity: 0.6, radius: 3 }
        : { x: s.k, y: s.p, label: 'sheet (tetrahedra)', color: PALETTE[5], points: true, width: 1, radius: 2 });
    }
    if (w.has('sheetp1')) {
      const s = e.pk('sheetp1', P, false);
      series.push({ x: s.k, y: s.p, label: dim === 3 ? 'sheet P1 (tetrahedra)' : 'sheet P1', color: P1_COLOR, points: true, width: 1.2, radius: 2.2 });
    }
    if (w.has('sheetx')) { const s = e.pk('sheetx', P, dw); series.push({ x: s.k, y: s.p, label: dim === 3 ? 'sheet exact P0 (tetrahedra)' : 'sheet exact P0', color: EXACT0_COLOR, points: true, width: 1.2, radius: 2.2 }); }
    if (w.has('sheetxp1')) { const s = e.pk('sheetxp1', P, dw); series.push({ x: s.k, y: s.p, label: dim === 3 ? 'sheet exact P1 (tetrahedra)' : 'sheet exact P1', color: EXACT1_COLOR, points: true, width: 1.2, radius: 2.2 }); }
    const notes = [], infos = [];
    for (const [id, p1, color, lab] of [['dir0', false, DIRECT0_COLOR, 'direct sheet spectrum (P0)'], ['dir1', true, DIRECT1_COLOR, 'direct sheet spectrum (P1)']]) {
      if (!w.has(id)) continue;
      const s = e.directPk(p1, P);
      if (!s) { notes.push(e.directNote(p1, P) || `direct sheet spectrum (${p1 ? 'P1' : 'P0'}) is not available yet.`); continue; }
      series.push({ x: s.k, y: s.p, label: lab, color, points: true, width: 1.2, radius: 2.6 });
      const i = e.directInfo[id];
      if (i) infos.push(`${lab}: ${i.path === 'GPU' ? 'GPU kernel' : 'WASM fallback'}, ${Math.round(i.nmodes).toLocaleString('en-US')} lattice modes (${i.perBin} per bin), ${i.ms >= 1000 ? (i.ms / 1000).toFixed(2) + ' s' : Math.round(i.ms) + ' ms'}`);
    }
    if (w.has('nufft')) {
      const ni = e.nufftInfo(P);
      if (!ni.ok) notes.push(ni.note);
      else { const s = e.pk('nufft', P, false); series.push({ x: s.k, y: s.p, label: `NUFFT density (refine ${P.rf})`, color: NUFFT_COLOR, points: true, width: 1.2, radius: 2.4 }); }
    }
    this.dirNote.hidden = notes.length === 0; this.dirNote.textContent = notes.join(' ');
    this.dirInfo.hidden = infos.length === 0; this.dirInfo.textContent = infos.join(' · ');
    this.dirCap.hidden = !(w.has('dir0') || w.has('dir1')); this.nufftCap.hidden = !w.has('nufft');
    if (w.has('cic')) {
      const s = e.pk('cic', P, true);
      series.push(dim === 2 ? { x: s.k, y: s.p, label: 'CIC (deconv.)', color: PALETTE[5], points: true, width: 1, radius: 2 }
        : { x: s.k, y: s.p, label: primLabel, color: PALETTE[0], points: true, width: 3, opacity: 0.6, radius: 3 });
    }
    if (w.has('hc')) { const s = e.pk('hc', P); series.push({ x: s.k, y: s.p, label: 'Hopf–Cole' + hcTag, color: PALETTE[1], points: true, width: 1.2, radius: 2 }); }

    if (w.has('hcdual')) { const s = e.pk('hcdual', P, dw); series.push({ x: s.k, y: s.p, label: 'Hopf–Cole dual sheet' + hcTag, color: DUAL_COLOR, points: true, width: 1.2, radius: 2 }); }

    let cs2 = null, fitNote = '';
    const spt = e.gaussian && !noLoop ? (w.has('spt') || w.has('eft') || w.has('p22') || w.has('p13') ? e.loop(P, 0) : null) : null;
    if (spt) {
      const tot = Float64Array.from(spt.k, (_, i) => spt.plin[i] + spt.p22[i] + spt.p13[i]);
      if (w.has('spt')) series.push({ x: spt.k, y: tot, label: '1-loop SPT', color: PALETTE[2], width: 1.8 });
      if (w.has('p22')) series.push({ x: spt.k, y: spt.p22, label: 'P22', color: PALETTE[2], width: 1.2, dash: '5 3' });
      if (w.has('p13')) series.push({ x: spt.k, y: Float64Array.from(spt.p13, Math.abs), label: '|P13|', color: PALETTE[3], width: 1.2, dash: '5 3' });
      if (w.has('eft')) {
        const meas = e.pk(prim, P, prim === 'cic');
        const m = Math.min(spt.k.length, meas.k.length);
        const km = S.km;
        const fit = fitCounterterm(spt.k.subarray(0, m), meas.p.subarray(0, m), tot.subarray(0, m), spt.plin.subarray(0, m), km);
        cs2 = fit.cs2;
        const eft = Float64Array.from(spt.k, (k, i) => tot[i] - 2 * cs2 * k * k * spt.plin[i]);
        series.push({ x: spt.k, y: eft, label: `EFT fit (c_s²=${fmtNum(cs2, 2)})`, color: PALETTE[4], width: 2.2 });
      }
    }
    if (e.gaussian && !noLoop && w.has('za')) {
      const za = e.loop(P, 1);
      series.push({ x: za.k, y: Float64Array.from(za.k, (_, i) => za.plin[i] + za.p22[i] + za.p13[i]), label: '1-loop Zel’dovich', color: PALETTE[3], width: 1.6 });
    }
    this.cs2 = cs2;
    this.csOut.set(cs2 === null ? (w.has('eft') && !e.gaussian ? 'n/a (needs Gaussian IC)' : '–') : fmtNum(cs2, 3));

    // y-range: keep ~10 decades below the top
    let top = 0;
    for (const s of series) for (let i = 0; i < s.y.length; i++) { const v = s.y[i]; if (v > top && Number.isFinite(v)) top = v; }
    const ylim = top > 0 ? [top * 1e-10, top * 2] : undefined;
    const pk = this.pkPlot;
    pk.setAxes({ xlog: true, ylog: true, xlabel: 'k  [rad / L]', ylabel: `P(k)  [L^${dim}]`, xlim: [kminPlot, kmaxPlot], ylim });
    pk.setSeries(series);
    const markers = [{ x: e.knyq, label: 'k_Nyq', color: '#888' }];
    if (P.R > 0) markers.push({ x: 1 / P.R, label: '1/R', color: '#aa66cc' });
    if (w.has('eft') && cs2 !== null) markers.push({ x: S.km, label: 'k_max', color: PALETTE[4] });
    pk.setMarkers(markers);
    pk.draw();

    // ---- r(k)
    const rs = [];
    const rp = e.rk(prim, P), rh = e.rk('hc', P);
    rs.push({ x: rp.k, y: rp.r, label: `${dim === 2 ? 'sheet' : 'CIC'} × linear`, color: PALETTE[0], points: true, width: 2.4, opacity: 0.7, radius: 2.5 });
    rs.push({ x: rh.k, y: rh.r, label: 'Hopf–Cole' + hcTag + ' × linear', color: PALETTE[1], points: true, width: 1.2, radius: 2 });
    if (w.has('sheetp1')) { const r1 = e.rk('sheetp1', P); rs.push({ x: r1.k, y: r1.r, label: `${dim === 2 ? 'sheet' : 'tetrahedral sheet'} P1 × linear`, color: P1_COLOR, points: true, width: 1.2, radius: 2.2 }); }
    if (w.has('sheetx')) { const rx = e.rk('sheetx', P); rs.push({ x: rx.k, y: rx.r, label: `${dim === 2 ? 'sheet' : 'tetrahedral sheet'} exact P0 × linear`, color: EXACT0_COLOR, points: true, width: 1.2, radius: 2.2 }); }
    if (w.has('sheetxp1')) { const rx = e.rk('sheetxp1', P); rs.push({ x: rx.k, y: rx.r, label: `${dim === 2 ? 'sheet' : 'tetrahedral sheet'} exact P1 × linear`, color: EXACT1_COLOR, points: true, width: 1.2, radius: 2.2 }); }
    if (w.has('nufft') && e.nufftInfo(P).ok) { const rx = e.rk('nufft', P); rs.push({ x: rx.k, y: rx.r, label: `NUFFT density (refine ${P.rf}) × linear`, color: NUFFT_COLOR, points: true, width: 1.2, radius: 2.4 }); }
    if (w.has('hcdual')) { const rd = e.rk('hcdual', P); rs.push({ x: rd.k, y: rd.r, label: 'Hopf–Cole dual sheet' + hcTag + ' × linear', color: DUAL_COLOR, points: true, width: 1.2, radius: 2 }); }
    const sv2 = e.sigmaV2();
    if (sv2 !== null) {
      const ks = Float64Array.from({ length: 160 }, (_, i) => kminPlot * Math.pow(kmaxPlot / kminPlot, i / 159));
      rs.push({ x: ks, y: Float64Array.from(ks, (k) => Math.exp(-0.5 * k * k * sv2 * P.D * P.D)), label: 'Zel’dovich propagator', color: '#8a8f98', dash: '5 3', width: 1.8 });
    }
    this.rPlot.setAxes({ xlog: true, ylog: false, xlabel: 'k  [rad / L]', ylabel: 'r(k)', xlim: [kminPlot, kmaxPlot], ylim: [0, 1.05] });
    this.rPlot.setSeries(rs);
    this.rPlot.setMarkers([{ x: e.knyq, label: 'k_Nyq', color: '#888' }, ...(P.R > 0 ? [{ x: 1 / P.R, label: '1/R', color: '#aa66cc' }] : [])]);
    this.rPlot.draw();
    this.drawPhase(P, hcTag);
    this.markStale(false);
  }

  /** Series of the phase-sum plot: [id, label, colour] for the selected field series (same colours as P(k)). */
  phaseSeries(P, hcTag = '') {
    const e = this.app.eng, w = this.wants(), d3 = e.dim === 3, out = [];
    const add = (id, label, color) => { if (w.has(id)) out.push({ id, label, color }); };
    add('lin', 'linear theory', '#8a8f98');
    add('sheet', d3 ? 'sheet (tetrahedra)' : 'sheet', d3 ? PALETTE[5] : PALETTE[0]);
    add('sheetp1', d3 ? 'sheet P1 (tetrahedra)' : 'sheet P1', P1_COLOR);
    add('sheetx', d3 ? 'sheet exact P0 (tetrahedra)' : 'sheet exact P0', EXACT0_COLOR);
    add('sheetxp1', d3 ? 'sheet exact P1 (tetrahedra)' : 'sheet exact P1', EXACT1_COLOR);
    if (w.has('nufft') && e.nufftInfo(P).ok) add('nufft', `NUFFT density (refine ${P.rf})`, NUFFT_COLOR);
    add('cic', d3 ? 'CIC' : 'CIC', d3 ? PALETTE[0] : PALETTE[5]);
    add('hc', 'Hopf–Cole' + hcTag, PALETTE[1]);
    add('hcdual', 'Hopf–Cole dual sheet' + hcTag, DUAL_COLOR);
    return out;
  }

  /** Triangle statistics of one field: cached per (field, k_max, bins, samples, seed); one WASM call. */
  phaseOf(id, P) {
    const e = this.app.eng, S = this.app.S;
    // k_min = kf/2 only drops the mean mode (its phase is meaningless); k_max is S.psk in units of k_f, passed in rad/L
    return e.phaseSums(id, P, 0.5 * e.kf, S.psk * e.kf, PH_NBINS, e.dim === 3 ? 20000 : 40000, S.seed);
  }

  /** Task: fill the phase-sum cache for the selected fields. */
  async phaseSums(P) {
    const e = this.app.eng;
    for (const s of this.phaseSeries(P)) {
      if (s.id !== 'lin') await e.need('analysis', s.id, P);
      this.phaseOf(s.id, P);
    }
  }

  drawPhase(P, hcTag = '') {
    const S = this.app.S, e = this.app.eng, amp = S.psw === 'a';
    const list = this.phaseSeries(P, hcTag), series = [];
    this.phVals.textContent = '';
    const dth = 2 * Math.PI / PH_NBINS;
    const th = Float64Array.from({ length: PH_NBINS }, (_, i) => -Math.PI + (i + 0.5) * dth);
    let top = 1;
    for (const s of list) {
      const r = this.phaseOf(s.id, P);
      const y = Float64Array.from(amp ? r.wpdf : r.pdf, (v) => 2 * Math.PI * v);
      for (const v of y) if (v > top) top = v;
      series.push({ x: th, y, label: s.label, color: s.color, width: 1.8, opacity: 0.9 });
      const row = el('span', 'lab-phval', this.phVals);
      const sw = el('span', 'lab-phsw', row); sw.style.background = s.color;
      el('span', null, row, `${s.label}: ⟨cos θ⟩ = ${r.n > 0 ? (amp ? r.wcos : r.cos).toFixed(3) : '–'}`);
      row.title = `${Math.round(r.n).toLocaleString('en-US')} triangles`;
    }
    series.push({ x: Float64Array.of(-Math.PI, Math.PI), y: Float64Array.of(1, 1), label: 'Gaussian field', color: '#8a8f98', dash: '1 4', width: 1.8, line: true });
    this.phPlot.setAxes({ xlog: false, ylog: false, xlabel: 'θ = φ(k₁) + φ(k₂) − φ(k₁+k₂)  [rad]', ylabel: amp ? '2π p(θ), amplitude-weighted' : '2π p(θ)', xlim: [-Math.PI, Math.PI], ylim: [0, Math.max(2, Math.min(top * 1.08, 40))] });
    this.phPlot.setSeries(series);
    this.phPlot.setMarkers([]);
    this.phPlot.draw();
  }

  /** Update the numeric readouts (cheap; uses cached values only where expensive). */
  updateReadouts(P, { dsc, dsc1 }) {
    const e = this.app.eng;
    if (!e.sim) return;
    this.ro.sigma.set(fmtNum(e.sim.sigma0(), 3));
    this.ro.dsc.set(fmtNum(dsc, 3));
    this.ro.dsc1.set(fmtNum(dsc1, 3));
    const hc = this.app.hcIfCached(P);
    this.ro.nu.set(hc ? fmtNum(hc.nuEff, 3) : '–');
    this.ro.exp.set(hc ? fmtNum(hc.range, 3) + (P.me === 0 && hc.range > P.mx * 1.001 ? ' (>max)' : '') : '–');
    const n = e.n, d = e.dim;
    this.ro.modes.set(`${n}${d === 2 ? '²' : '³'} = ${Math.pow(n, d).toLocaleString('en-US')} (kf=${fmtNum(e.kf, 3)}, kNyq=${fmtNum(e.knyq, 3)})`);
  }

  destroy() { this.root.remove(); }
}
