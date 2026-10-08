// pdfs.js - the "PDFs" section: one-point probability density of the density field for every available method,
// at the grid scale and after top-hat smoothing (sphere in 3D, disc in 2D) of an adjustable diameter.
//
// Histograms are of x = log10(1+delta), 1+delta clamped to >= 1e-3, in NBIN fixed bins on [X0, X1] (the same bins for every method and
// scale, so the curves are comparable), normalised as a probability density in x: counts / (N_total dx).  Everything is cached in the
// Engine memo under (field key, diameter), so a D / nu / source change refreshes only the affected methods and a diameter change
// only re-smooths.  The smoothing itself is the Fourier-space top-hat: WASM CosmoSim.tophat_smooth, or the GPU FFT in 3D
// (GpuCosmo3D.smoothTophat) when the GPU compute path is active.

import { LinePlot, PALETTE } from '../viz/plot.js';
import { el, numIn, fmtNum } from './dom.js';
import { DUAL_COLOR, P1_COLOR, EXACT0_COLOR, EXACT1_COLOR } from './spectra.js';
import { Engine } from './engine.js';

export const NBIN = 60, X0 = -2, X1 = 2.5, DX = (X1 - X0) / NBIN;
const LN10 = Math.LN10, RHO_MIN = 1e-3;
const CENTERS = Float64Array.from({ length: NBIN }, (_, i) => X0 + (i + 0.5) * DX);
const YLIM = [1e-4, 10];

/** Histogram and moments of the overdensity field delta (Float32Array, any length). */
export function pdfStats(delta) {
  const N = delta.length;
  const cnt = new Float64Array(NBIN);
  let sD = 0, sL = 0, low = 0, maxR = -Infinity;
  for (let i = 0; i < N; i++) {
    const d = delta[i], r = 1 + d;
    sD += d;
    const rc = r > RHO_MIN ? r : RHO_MIN;
    const x = Math.log10(rc);
    sL += x;
    if (r < 0.1) low++;
    if (r > maxR) maxR = r;
    const b = Math.floor((x - X0) / DX);
    if (b >= 0 && b < NBIN) cnt[b]++;
  }
  const mean = sD / N, meanX = sL / N;
  let m2 = 0, m3 = 0, v2 = 0;
  for (let i = 0; i < N; i++) {
    const d = delta[i] - mean, r = 1 + delta[i];
    m2 += d * d; m3 += d * d * d;
    const x = Math.log10(r > RHO_MIN ? r : RHO_MIN) - meanX;
    v2 += x * x;
  }
  m2 /= N; m3 /= N; v2 /= N;
  const dens = new Float64Array(NBIN);
  for (let b = 0; b < NBIN; b++) dens[b] = cnt[b] / (N * DX);
  return {
    n: N, mean, varD: m2, skew: m2 > 0 ? m3 / Math.pow(m2, 1.5) : NaN,
    meanX, varX: v2,                    // mean and variance of log10(1+delta) (= those of ln(1+delta) / ln10, / ln10^2)
    varLn: v2 * LN10 * LN10, fLow: low / N, maxRho: maxR, dens,
  };
}

const gaussX = (x, mu, s2) => Math.exp(-0.5 * (x - mu) * (x - mu) / s2) / Math.sqrt(2 * Math.PI * s2);

const COLS = [
  ['σ²(δ)', 'variance of the overdensity δ'],
  ['σ²(ln(1+δ))', 'variance of ln(1+δ), with 1+δ clamped to ≥ 10⁻³'],
  ['skew(δ)', 'skewness of δ: third central moment / variance^(3/2)'],
  ['f(1+δ<0.1)', 'fraction of cells with 1+δ < 0.1'],
  ['max(1+δ)', 'largest value of 1+δ'],
];

export class PdfLab {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-pdfs', host);
    el('p', 'hcc-note lab-leg-intro', this.root,
      'One-point probability density of the density contrast for every method available in this mode, at two Eulerian smoothing scales: the raw grid cells, and the field smoothed with a top-hat (a sphere in 3D, a disc in 2D) of adjustable diameter. '
      + 'Histograms are of log₁₀(1+δ) in fixed bins so that the methods can be compared directly.');
    const bar = this.bar = el('div', 'hcc-controls lab-fbar', this.root);
    this.diam = numIn(bar, { label: 'diameter [cells]', value: S.pd, min: 2, max: 64, step: 1, width: '4rem',
      title: 'Diameter of the top-hat sphere (disc in 2D) in grid cells; radius = diameter / 2 cells. 2 to 64.',
      onChange: (v) => { v = Math.round(v); this.diam.set(v); S.pd = v; app.hashChanged(); app.runPdfs(); } });
    this.note = el('p', 'hcc-note lab-leg-status', this.root);
    this.note.hidden = true;

    const grid = el('div', 'hcc-grid2 lab-grid lab-pdf-grid', this.root);
    this.panels = [0, 1].map(() => {
      const root = el('div', 'hcc-panel lab-panel', grid);
      const title = el('div', 'hcc-title', root, '');
      const plot = new LinePlot(el('div', 'lab-plot', root), { width: 500, height: 360 });
      const cap = el('p', 'hcc-note lab-cap', root);
      const tw = el('div', 'lab-pdf-tablewrap', root);
      return { root, title, plot, cap, tw };
    });
    this.general = el('p', 'hcc-note lab-cap lab-pdf-general', this.root);
    this.general.innerHTML = [
      'Linear theory predicts a Gaussian δ; once its rms exceeds about 0.3 the Gaussian already assigns probability to 1+δ < 0 (negative densities), which a pressureless fluid cannot have, and the dotted grey curve (the Gaussian mapped to log₁₀(1+δ)) misses exactly that mass.',
      'Gravity skews the distribution: underdense regions empty out towards a void density (a pile-up of cells just above 1+δ = 0 in log₁₀(1+δ) ≈ −1), while collapse produces a long high-density tail.',
      'A lognormal (dashed grey: a Gaussian in log₁₀(1+δ) with the mean and variance of ln(1+δ) of the sheet in 2D, of CIC in 3D) is often a good approximation (Coles & Jones 1991), though it cannot capture the shape of the tail exactly.',
      'The P1 sheet gives every simplex a linear density shape (interpolating the vertex values 1/|J|) that still deposits exactly the simplex mass, so before shell crossing its PDF follows that of the plain sheet closely (the two differ only by cell-scale discretization noise), and it is the cleaner reference for the Hopf–Cole variants.',
      'The exact sheet deposits (P0 and P1) clip every simplex against the cells it overlaps and integrate its density over each piece, so they have no sampling noise at all: their grid-scale PDFs show only the real structure of the simplicial sheet, and are the cleanest reference for the other methods.',
      'The Hopf–Cole dual sheet is the mass-conserving density of the Hopf–Cole inverse map (cell mass = Lagrangian volume of its preimage); before shell crossing it should follow the sheet closely in both the voids and the high-density tail, where the finite-difference Hopf–Cole density under-resolves the peaks.',
      'After shell crossing the multi-stream sheet adds the streams, whereas the Hopf–Cole (adhesion) solution keeps one stream and glues them into walls; the two therefore populate the high-density tail differently, and CIC adds its own smoothing of the sheet.',
      'Top-hat smoothing lowers the variance and Gaussianises the PDF (averaging many cells), so the smoothed curves lie closer to the Gaussian and the lognormal, and the rare high values of the grid-scale PDF disappear.',
      'The top-hat window rings in Fourier space (it has negative lobes), so the smoothed density can dip slightly below the true minimum, even to 1+δ slightly below zero in deep voids.',
    ].join(' ');
    this.lastP = null;
    this.syncMode();
  }

  syncMode() { this.diam.set(this.app.S.pd); }

  markStale(b) { for (const p of this.panels) p.root.classList.toggle('is-stale', b); }

  // ---------------------------------------------------------------- method list

  /** Include the sheet? 2D: always. 3D: when the GPU path computes it (milliseconds), at n <= 64, or if it is already cached. */
  includeSheet(P) {
    const e = this.app.eng;
    return P.mode === 2 || e.gpuActive(P) || P.n <= 64 || e.peek(['sheet', P.D, P.order]) !== undefined || e.peek(['sheetp1', P.D, P.order]) !== undefined;
  }

  /** Include the exact (clipped) sheet deposits? 2D: always (GPU or WASM, a fraction of a second). 3D: on the GPU path, at n <= 32, or if already cached. */
  includeExact(P) {
    const e = this.app.eng;
    return P.mode === 2 || e.gpuActive(P) || P.n <= 32 || e.peek(['sheetx', P.D, P.order]) !== undefined || e.peek(['sheetxp1', P.D, P.order]) !== undefined;
  }

  /** Method ids in plot order. Zel'dovich / Legendre are left out when they coincide with the selected source. */
  methods(P, runtime = true) {
    const m = [];
    if (P.mode === 2 || !runtime || this.includeSheet(P)) m.push('sheet', 'sheetp1');
    if (P.mode === 2 || !runtime || this.includeExact(P)) m.push('sheetx', 'sheetxp1');
    m.push('cic', 'hc', 'hcdual');
    if (P.hs !== 'zel') m.push('hcz'); else m.push('hcl');
    m.push('lin');
    return m;
  }

  style(m, P) {
    const d3 = P.mode === 3;
    const hcTag = P.hs === 'zel' || !P.hs ? ' (Zel’dovich)' : P.hs === 'lptT' && P.order > 2 ? ` (Legendre ${P.order}LPT+T)` : ` (Legendre ${P.order}LPT)`;
    switch (m) {
      case 'lin': return { label: 'linear D·δ₀', color: '#8a8f98', width: 2.2 };
      case 'sheet': return d3 ? { label: 'sheet (tetrahedra)', color: PALETTE[5], width: 1.4 } : { label: 'sheet', color: PALETTE[0], width: 3, opacity: 0.6 };
      case 'sheetp1': return { label: d3 ? 'sheet P1 (tetrahedra)' : 'sheet P1', color: P1_COLOR, width: 1.4 };
      case 'sheetx': return { label: d3 ? 'sheet exact P0 (tetrahedra)' : 'sheet exact P0', color: EXACT0_COLOR, width: 1.4 };
      case 'sheetxp1': return { label: d3 ? 'sheet exact P1 (tetrahedra)' : 'sheet exact P1', color: EXACT1_COLOR, width: 1.4 };
      case 'cic': return d3 ? { label: 'CIC', color: PALETTE[0], width: 3, opacity: 0.6 } : { label: 'CIC', color: PALETTE[5], width: 1.4 };
      case 'hc': return { label: 'Hopf–Cole' + hcTag, color: PALETTE[1], width: 1.4 };
      case 'hcdual': return { label: 'Hopf–Cole dual sheet' + hcTag, color: DUAL_COLOR, width: 1.4 };
      case 'hcz': return { label: 'Zel’dovich Hopf–Cole', color: PALETTE[3], width: 1.4 };
      case 'hcl': return { label: 'Legendre (nLPT) Hopf–Cole', color: PALETTE[2], width: 1.4 };
      default: return { label: m, color: PALETTE[7], width: 1.4 };
    }
  }

  // ---------------------------------------------------------------- computation

  /** Cached statistics of method `which` at diameter `diam` cells (0 = grid scale). */
  stats(which, P, diam) {
    const e = this.app.eng;
    return e.memo(['pdf', ...e.fieldKey(which, P), diam], () => e.time(diam ? 'PDF (smoothed)' : 'PDF', () => pdfStats(diam ? e.smooth(which, P, diam) : e.delta(which, P))));
  }

  /** hcOnly: only the Hopf-Cole variants (the fast nu / source path; everything else is cached). */
  tasks(P, hcOnly = false) {
    const e = this.app.eng, d3 = P.mode === 3;
    const t = [];
    for (const m of this.methods(P, false)) {
      const hc = Engine.isHcAny(m);
      if (hcOnly && !hc) continue;
      t.push({
        label: `PDFs: ${m === 'lin' ? 'linear' : m === 'cic' ? 'CIC' : m === 'sheet' ? 'sheet' : m === 'sheetp1' ? 'sheet P1' : m === 'sheetx' ? 'sheet exact P0' : m === 'sheetxp1' ? 'sheet exact P1' : m === 'hcdual' ? 'Hopf–Cole dual sheet' : 'Hopf–Cole'}${m === 'hcz' ? ' (Zel’dovich)' : m === 'hcl' ? ' (Legendre)' : ''}`,
        heavy: d3 && (m === 'sheet' || m === 'sheetp1' || Engine.isExact(m)) && !e.gpuActive(P),
        fn: async () => {
          if ((m === 'sheet' || m === 'sheetp1') && !this.includeSheet(P)) return;
          if (Engine.isExact(m) && !this.includeExact(P)) return;
          if (m !== 'lin') await e.need('field', m, P);
          this.stats(m, P, 0);
          await e.needSmooth(m, P, P.pd);
          this.stats(m, P, P.pd);
        },
      });
    }
    t.push({ label: 'PDFs: draw', fn: () => this.draw(P) });
    return t;
  }

  draw(P) {
    const e = this.app.eng;
    if (!e.sim || e.dim !== P.mode) return;
    this.lastP = P;
    const ms = this.methods(P);
    const d3 = P.mode === 3;
    const refM = d3 ? 'cic' : 'sheet';
    const skipped = d3 && !ms.includes('sheet');
    const skippedX = d3 && !ms.includes('sheetx');
    this.note.hidden = !(skipped || skippedX);
    this.note.textContent = [
      skipped ? `The tetrahedral sheet is not shown: at ${P.n}³ without the GPU compute path it takes tens of seconds in WASM (shown at 64³ or below, with GPU compute, or once it has been computed in the Legendre lab).` : '',
      skippedX ? `The exact (clipped) tetrahedral sheets are not shown: without the GPU compute path they take several seconds in WASM even at 64³ (shown with GPU compute, at 32³ or below, or once computed in the Legendre lab).` : '',
    ].filter(Boolean).join(' ');
    const scales = [0, P.pd];
    this.diam.set(P.pd);
    scales.forEach((diam, idx) => {
      const pan = this.panels[idx];
      const R = diam / 2;
      pan.title.textContent = diam ? `top-hat, ⌀ ${diam} cells` : 'grid cells';
      const series = [];
      const rows = [];
      for (const m of ms) {
        const st = this.stats(m, P, diam);
        const sty = this.style(m, P);
        series.push({ x: CENTERS, y: st.dens, ...sty });
        rows.push({ m, sty, st });
      }
      // reference curves
      const lin = this.stats('lin', P, diam), ref = this.stats(refM, P, diam);
      const xs = Float64Array.from({ length: 241 }, (_, i) => X0 + (X1 - X0) * i / 240);
      let gaussNote = '';
      if (lin.varD > 0) {
        const s2 = lin.varD, mu = lin.mean;
        series.push({ x: xs, y: Float64Array.from(xs, (x) => { const r = Math.pow(10, x), d = r - 1; return Math.exp(-0.5 * (d - mu) * (d - mu) / s2) / Math.sqrt(2 * Math.PI * s2) * r * LN10; }),
          label: 'Gaussian (linear σ)', color: '#8a8f98', width: 1.6, dash: '2 3' });
        gaussNote = `Gaussian: σ(δ_lin) = ${fmtNum(Math.sqrt(s2), 3)}`;
      }
      let lnNote = '';
      if (ref.varX > 0) {
        series.push({ x: xs, y: Float64Array.from(xs, (x) => gaussX(x, ref.meanX, ref.varX)), label: `lognormal (${refM === 'cic' ? 'CIC' : 'sheet'})`, color: '#8a8f98', width: 1.6, dash: '7 3' });
        lnNote = `lognormal: ⟨ln(1+δ)⟩ = ${fmtNum(ref.meanX * LN10, 3)}, σ² = ${fmtNum(ref.varLn, 3)}`;
      }
      pan.plot.setAxes({ xlog: false, ylog: true, xlabel: 'log₁₀(1+δ)', ylabel: 'p(log₁₀(1+δ))', xlim: [X0, X1], ylim: YLIM });
      pan.plot.setSeries(series);
      pan.plot.setMarkers([]);
      pan.plot.draw();
      pan.cap.textContent = (diam
        ? `Smoothed with a top-hat of radius ${R} cells (${fmtNum(R / P.n, 3)} L, diameter ${diam} cells = ${fmtNum(diam / P.n, 3)} L) in Fourier space. `
        : 'Raw values of 1+δ in the cells of the grid. ')
        + [gaussNote, lnNote].filter(Boolean).join('; ') + '.';
      this.fillTable(pan.tw, rows);
    });
    this.markStale(false);
  }

  fillTable(host, rows) {
    host.textContent = '';
    const tb = el('table', 'lab-pdf-table', host);
    const hr = el('tr', null, el('thead', null, tb));
    el('th', null, hr, 'method');
    for (const [c, tip] of COLS) { const th = el('th', null, hr, c); th.title = tip; }
    const body = el('tbody', null, tb);
    for (const { sty, st } of rows) {
      const tr = el('tr', null, body);
      const td = el('td', 'lab-pdf-name', tr);
      const sw = el('span', 'lab-pdf-swatch', td); sw.style.background = sty.color;
      el('span', null, td, sty.label);
      for (const v of [st.varD, st.varLn, st.skew, st.fLow, st.maxRho]) el('td', null, tr, fmtNum(v, 3));
    }
  }

  destroy() { for (const p of this.panels) p.plot.destroy(); this.root.remove(); }
}
