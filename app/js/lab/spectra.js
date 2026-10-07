// spectra.js - the "Spectra" section: P(k), r(k), EFT counterterm fit and readouts.

import { LinePlot, PALETTE } from '../viz/plot.js';
import { slider, checkbox, readout } from '../viz/ui.js';
import { fitCounterterm } from '../hcc.js';
import { el, fmtNum } from './dom.js';

const SERIES = [
  ['lin', 'linear theory'],
  ['sheet', 'sheet (measured)'],
  ['cic', 'CIC (deconv.)'],
  ['hc', 'Hopf–Cole (selected source)'],
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
    this.pkHost = el('div', 'lab-plot', pp);
    this.pkPlot = new LinePlot(this.pkHost, { width: 500, height: 380 });
    el('p', 'hcc-note lab-cap', pp, 'Dashed vertical lines mark k_Nyq and the smoothing scale 1/R. Linear theory is the smooth curve D²P0(k); the 1-loop curves are only available for Gaussian ICs. The EFT curve subtracts 2c_s²k²P_lin from the SPT 1-loop with c_s² fitted to the measured spectrum for k < k_max.');

    // ---- r(k) panel
    const rp = this.rPanel = el('div', 'hcc-panel lab-panel', grid);
    el('div', 'hcc-title', rp, 'Cross-correlation with the linear field r(k)');
    this.rHost = el('div', 'lab-plot', rp);
    this.rPlot = new LinePlot(this.rHost, { width: 500, height: 380 });
    el('p', 'hcc-note lab-cap', rp, 'r(k) = P_{f,lin} / √(P_f P_lin) measures how much of the evolved field’s phase information still matches linear theory. The dashed curve is the Zel’dovich propagator exp(−k²σ_Ψ²D²/2), with σ_Ψ² the per-axis displacement variance of the grid modes (Gaussian ICs only). It equals r(k) only while the evolved power is still close to linear, and is only approximate once streams cross.');

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
    this.checks.cic.el.querySelector('.hcc-label').textContent = d3 ? 'CIC (measured, deconv.)' : 'CIC (deconv.)';
    const ser = this.app.S.ser;
    for (const id of Object.keys(this.checks)) this.checks[id].set(ser.includes(id));
  }

  markStale(b) { this.pPanel.classList.toggle('is-stale', b); this.rPanel.classList.toggle('is-stale', b); }

  wants() {
    const w = new Set(this.app.S.ser);
    return w;
  }

  /** Tasks that fill the caches and finally draw both plots. */
  tasks(P) {
    const e = this.app.eng, w = this.wants();
    const dim = P.mode, gaussian = P.ic === 'g';      // engine state may not be configured yet when tasks are built
    const prim = dim === 2 ? 'sheet' : 'cic';
    const t = [];
    t.push({ label: 'P(k) linear', fn: () => { e.pk('lin', P); e.plinFine(P); } });
    // 3D GPU path: `need` runs the GPU FFT once per field and seeds P(k), r(k) and the Fourier maps
    if (w.has('sheet')) t.push({ label: dim === 3 ? 'P(k) sheet (tetrahedra)' : 'P(k) sheet', fn: async () => { await e.need('analysis', 'sheet', P); e.pk('sheet', P, false); } });
    if (w.has('cic')) t.push({ label: 'P(k) CIC', fn: async () => { await e.need('analysis', 'cic', P); e.pk('cic', P, true); } });
    if (w.has('hc')) t.push({ label: 'P(k) Hopf–Cole', fn: async () => { await e.need('analysis', 'hc', P); e.pk('hc', P); } });
    t.push({ label: 'r(k)', fn: async () => { await e.need('analysis', prim, P); await e.need('analysis', 'hc', P); e.rk(prim, P); e.rk('hc', P); e.sigmaV2(); } });
    const loops = gaussian && (w.has('spt') || w.has('eft') || w.has('p22') || w.has('p13') || w.has('za'));
    // lite: show the measured spectra first, the 1-loop curves (0.3 s in WASM) follow in a second drawing; only when this section is visible
    if (this.app.lite && loops && this.app.visible('s') && !e.peek(['loop1', 0])) t.push({ label: 'plots (1-loop follows)', fn: () => this.draw(P, true) });
    if (gaussian && (w.has('spt') || w.has('eft') || w.has('p22') || w.has('p13'))) {
      t.push({ label: 'SPT 1-loop', heavy: true, fn: () => e.loop(P, 0) });
    }
    if (gaussian && w.has('za')) t.push({ label: 'Zel’dovich 1-loop', heavy: true, fn: () => e.loop(P, 1) });
    t.push({ label: 'plots', fn: () => this.draw(P) });
    return t;
  }

  drawIfReady() { if (this.app.eng.sim && this.lastP) this.draw(this.lastP); }

  draw(P, noLoop = false) {
    const e = this.app.eng, S = this.app.S, w = this.wants();
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
    if (w.has('cic')) {
      const s = e.pk('cic', P, true);
      series.push(dim === 2 ? { x: s.k, y: s.p, label: 'CIC (deconv.)', color: PALETTE[5], points: true, width: 1, radius: 2 }
        : { x: s.k, y: s.p, label: primLabel, color: PALETTE[0], points: true, width: 3, opacity: 0.6, radius: 3 });
    }
    if (w.has('hc')) { const s = e.pk('hc', P); series.push({ x: s.k, y: s.p, label: 'Hopf–Cole' + hcTag, color: PALETTE[1], points: true, width: 1.2, radius: 2 }); }

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
    const sv2 = e.sigmaV2();
    if (sv2 !== null) {
      const ks = Float64Array.from({ length: 160 }, (_, i) => kminPlot * Math.pow(kmaxPlot / kminPlot, i / 159));
      rs.push({ x: ks, y: Float64Array.from(ks, (k) => Math.exp(-0.5 * k * k * sv2 * P.D * P.D)), label: 'Zel’dovich propagator', color: '#8a8f98', dash: '5 3', width: 1.8 });
    }
    this.rPlot.setAxes({ xlog: true, ylog: false, xlabel: 'k  [rad / L]', ylabel: 'r(k)', xlim: [kminPlot, kmaxPlot], ylim: [0, 1.05] });
    this.rPlot.setSeries(rs);
    this.rPlot.setMarkers([{ x: e.knyq, label: 'k_Nyq', color: '#888' }, ...(P.R > 0 ? [{ x: 1 / P.R, label: '1/R', color: '#aa66cc' }] : [])]);
    this.rPlot.draw();
    this.markStale(false);
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
