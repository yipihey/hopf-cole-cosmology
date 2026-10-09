// kernels.js - the "Kernels" section: the two-plane-wave check of the second-order mode-coupling kernels.
//
// For the initial condition delta0 = A1 cos(k1.x + phi1) + A2 cos(k2.x + phi2) the evolved density has, to second order, harmonics at
//   k1+k2:  c = 2 D^2 F2(k1,k2)  c0(k1) c0(k2)          k1-k2:  c = 2 D^2 F2(k1,-k2) c0(k1) c0(k2)*
//   2k1:    c = D^2 F2(k1,k1) c0(k1)^2                  2k2:    c = D^2 F2(k2,k2) c0(k2)^2
// (c = complex Fourier coefficient, field = sum_k c(k) e^{ik.x}) and, at first order, c(k1) = D g(k1) c0(k1) with g = 1 or exp(-nu k^2 D).
// The section MEASURES the complex amplitudes (WASM CosmoSim.mode_amplitudes; the linear c0 from the engine's linear field at D = 1, so the
// rescaling of the amplitudes and the smoothing R are accounted for), divides them by the products above and compares the real part with the
// analytic kernels (CosmoSim.f2_kernel): viscous F2^nu(nu, D) (kind 2), Zel'dovich F2^ZA (1) and gravitational EdS F2 (0).
// The Hopf-Cole column uses the Engine's 'hc' field, the sheet column the P1 sheet at the current LPT order.  The sweep plot re-solves the
// Hopf-Cole problem directly in the WASM sim along D or nu; the Engine's cache stores copies of its own solutions, so those direct solves do
// not invalidate anything (nothing in the lab reads the sim's "last Hopf-Cole run" lazily).

import { LinePlot, PALETTE } from '../viz/plot.js';
import { el, sel, fmtNum, tick, docLink } from './dom.js';
import { BOOK } from './catalog.js';
import { P1_COLOR } from './spectra.js';

const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cconj = (a) => [a[0], -a[1]];
const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
const cabs = (a) => Math.hypot(a[0], a[1]);
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const veq = (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
const vzero = (a) => veq(a, [0, 0, 0]);
const vstr = (a, dim) => '(' + a.slice(0, dim).join(', ') + ')';
const logspace = (a, b, n) => Float64Array.from({ length: n }, (_, i) => a * Math.pow(b / a, n > 1 ? i / (n - 1) : 0));
const C0_MIN = 1e-6;

// harmonic rows: [id, label, multiplicity of the sum in delta_2]
const ROWS = [
  { id: 'k1', label: 'k₁' },
  { id: 'sum', label: 'k₁+k₂', mult: 2 },
  { id: 'dif', label: 'k₁−k₂', mult: 2 },
  { id: 'd1', label: '2k₁', mult: 1 },
  { id: 'd2', label: '2k₂', mult: 1 },
];
const HCOL = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]];     // colours of the four harmonics (rows 1..4)

export class KernelLab {
  constructor(app, host) {
    this.app = app;
    const S = app.S;
    this.root = el('div', 'lab-kernels', host);
    const intro = el('p', 'hcc-note lab-leg-intro', this.root,
      'The two-plane-wave kernel check. For δ₀ = A₁cos(k₁·x+φ₁) + A₂cos(k₂·x+φ₂) the second-order density has harmonics at k₁±k₂, 2k₁ and 2k₂ whose amplitude, divided by '
      + 'D²c₀(k₁)c₀(k₂) (times the multiplicity), is the mode-coupling kernel F₂ of the dynamics. The table measures it (the complex amplitudes are taken from the fields themselves) '
      + 'for the Hopf–Cole density and for the sheet, and compares with the viscous kernel F₂^ν of the adhesion model, the Zel’dovich kernel F₂^ZA and the gravitational kernel F₂. '
      + 'Valid for D well below the shell-crossing time, where the third and higher orders (relative size ∝ D²) are small.');
    docLink(intro, BOOK.kernels);
    this.noIc = el('div', 'lab-kern-noic', this.root);
    el('p', 'hcc-note', this.noIc, 'The kernel check needs the two-plane-wave initial condition.');
    const b = el('button', 'hcc-btn', this.noIc, 'Switch to two plane waves'); b.type = 'button';
    b.addEventListener('click', () => this.useWaves());
    this.main = el('div', 'lab-kern-main', this.root);
    const bar = this.bar = el('div', 'hcc-controls lab-fbar', this.main);
    this.kxSel = sel(bar, { label: 'sweep', options: [['D', 'D'], ['nu', 'ν']], value: S.kx,
      title: 'Variable of the sweep plot: the growth factor D (from 0.02 D_sc to the current D, at the current ν) or the viscosity ν (1e-5 … 1e-2, at the current D).',
      onChange: (v) => { S.kx = v; app.hashChanged(); app.runKernels(); } });
    this.note = el('p', 'hcc-note lab-leg-status', this.root);
    this.note.hidden = true;
    const grid = el('div', 'hcc-grid2 lab-grid lab-kern-grid', this.main);
    this.tPanel = el('div', 'hcc-panel lab-panel', grid);
    this.tTitle = el('div', 'hcc-title', this.tPanel, 'Measured and analytic kernels');
    this.tw = el('div', 'lab-pdf-tablewrap', this.tPanel);
    this.readout = el('p', 'hcc-note lab-cap lab-kern-readout', this.tPanel);
    this.pPanel = el('div', 'hcc-panel lab-panel', grid);
    this.pTitle = el('div', 'hcc-title', this.pPanel, 'Kernel sweep');
    this.plot = new LinePlot(el('div', 'lab-plot', this.pPanel), { width: 520, height: 380 });
    this.cap = el('p', 'hcc-note lab-cap', this.pPanel);
    this.general = el('p', 'hcc-note lab-cap lab-pdf-general', this.main);
    this.general.textContent = 'Reading the table: the Hopf–Cole density (Burgers / adhesion dynamics with D as time) follows F₂^ν, which tends to the Zel’dovich kernel as ν → 0 and is damped at larger νk²D; '
      + 'it lacks the tidal term of gravity (F₂ − F₂^ZA = (3/14)[1 − (k̂₁·k̂₂)²]). The sheet at LPT order 1 follows F₂^ZA; at order ≥ 2 it follows the gravitational F₂. '
      + 'For parallel waves the tidal term vanishes and F₂^ZA = F₂. The imaginary part of the measured kernel (small grey number) should be ≈ 0. '
      + 'The analytic viscous kernel uses the effective viscosity ν_eff the solver actually applied (ν_eff ≥ ν: the solver raises ν so that the log-domain exponent range stays below the limit).';
    this.res = null;
    this.syncMode();
  }

  syncMode() { this.kxSel.set(this.app.S.kx); this.res = null; }

  markStale(b) { this.tPanel.classList.toggle('is-stale', b); this.pPanel.classList.toggle('is-stale', b); }

  /** The "switch to two plane waves" button: same path as the IC select of the controls. */
  useWaves() {
    const app = this.app, C = app.controls;
    if (C && C.ic) C.ic.set('w');
    app.setParam('ic', 'w', 'ic');
    if (C && C.syncVisibility) C.syncVisibility();
  }

  // ---------------------------------------------------------------- geometry

  /** Mode vectors of the harmonics for the applied state P (z = 0 in 2D). */
  geometry(P) {
    const dim = P.mode;
    const v = (k) => [k[0], k[1], dim === 3 ? k[2] : 0];
    const k1 = v(P.k1), k2 = v(P.k2);
    const m2 = vscale(k2, -1);
    const rows = [
      { ...ROWS[0], vec: k1 },
      { ...ROWS[1], vec: vadd(k1, k2), a: k1, b: k2 },
      { ...ROWS[2], vec: vadd(k1, m2), a: k1, b: m2 },
      { ...ROWS[3], vec: vscale(k1, 2), a: k1, b: k1 },
      { ...ROWS[4], vec: vscale(k2, 2), a: k2, b: k2 },
    ];
    // degenerate: zero vector, or a harmonic that coincides (up to sign) with k1, k2 or another harmonic: the amplitudes then mix
    const sameMode = (a, b) => veq(a, b) || veq(a, vscale(b, -1));
    rows.forEach((r, i) => {
      r.bad = '';
      if (i > 0 && vzero(r.vec)) r.bad = 'zero mode';
      else if (i > 0 && (sameMode(r.vec, k1) || sameMode(r.vec, k2))) r.bad = 'coincides with k₁ or k₂';
      else if (i > 0) for (let j = 1; j < rows.length; j++) if (j !== i && sameMode(r.vec, rows[j].vec)) r.bad = `coincides with ${rows[j].label}`;
      if (i === 0 && (vzero(k1) || sameMode(k1, k2))) r.bad = sameMode(k1, k2) ? 'k₂ = ±k₁' : 'zero mode';
    });
    const cos = vdot(k1, k2) / Math.sqrt(vdot(k1, k1) * vdot(k2, k2));
    return { dim, k1, k2, rows, cos, ok: !(vzero(k1) || vzero(k2)) };
  }

  /** Kernel values [{re, im, ok}] for rows 0..4 from the complex amplitudes amp[i] of the field at rows[i].vec; c1, c2 = linear c0(k1), c0(k2). */
  kernelsFrom(geo, amp, c1, c2, D) {
    const bad0 = cabs(c1) < C0_MIN || cabs(c2) < C0_MIN || !(D > 0);
    const out = [];
    geo.rows.forEach((r, i) => {
      const c = amp[i];
      if (r.bad || bad0 || !c || !Number.isFinite(c[0]) || !Number.isFinite(c[1])) { out.push({ re: NaN, im: NaN, ok: false }); return; }
      let K;
      if (i === 0) K = cdiv(c, [D * c1[0], D * c1[1]]);
      else {
        const prod = i === 1 ? cmul(c1, c2) : i === 2 ? cmul(c1, cconj(c2)) : i === 3 ? cmul(c1, c1) : cmul(c2, c2);
        K = cdiv(c, [r.mult * D * D * prod[0], r.mult * D * D * prod[1]]);
      }
      out.push({ re: K[0], im: K[1], ok: Number.isFinite(K[0]) });
    });
    return out;
  }

  /** Complex amplitudes of `field` at the given mode vectors. */
  amps(field, vecs) {
    const a = this.app.eng.sim.mode_amplitudes(field, Int32Array.from(vecs.flat()));
    return vecs.map((_, i) => [a[2 * i], a[2 * i + 1]]);
  }

  /** Linear c0 (D = 1) of the two waves, measured from the engine's linear field (includes the amplitude rescaling and smoothing). */
  linearAmps(P, geo) {
    const e = this.app.eng;
    const lin = e.delta('lin', { ...P, D: 1 });
    const [c1, c2] = this.amps(lin, [geo.k1, geo.k2]);
    return { c1, c2 };
  }

  // ---------------------------------------------------------------- analytic kernels

  f2(kind, a, b, nu, d) { return this.app.eng.sim.f2_kernel(kind, a[0], a[1], a[2], b[0], b[1], b[2], nu, d); }

  /** Analytic [viscous, ZA, gravity] for row r (first-order propagators for the k1 row). */
  analytic(geo, i, nu, D) {
    const r = geo.rows[i];
    if (i === 0) {
      const kf = this.app.eng.kf, k2 = kf * kf * vdot(r.vec, r.vec);
      return [Math.exp(-nu * k2 * D), 1, 1];
    }
    return [this.f2(2, r.a, r.b, nu, D), this.f2(1, r.a, r.b, nu, D), this.f2(0, r.a, r.b, nu, D)];
  }

  // ---------------------------------------------------------------- computation

  sheetAvailable(P) {
    const e = this.app.eng;
    return P.mode === 2 || e.gpuActive(P) || P.n <= 64 || e.peek(['sheetp1', P.D, P.order]) !== undefined;
  }

  tasks(P) {
    const app = this.app, e = app.eng;
    if (P.ic !== 'w') return [{ label: 'Kernels: draw', fn: () => this.draw(P) }];
    const t = [];
    t.push({
      label: 'Kernels: measure modes',
      fn: async () => {
        const geo = this.geometry(P);
        await e.need('field', 'hc', P);
        const hc = e.delta('hc', P);
        const hcInfo = e.hcCached(P);
        const { c1, c2 } = this.linearAmps(P, geo);
        const vecs = geo.rows.map((r) => r.vec);
        const res = { P, geo, c1, c2, nuEff: hcInfo && Number.isFinite(hcInfo.nuEff) ? hcInfo.nuEff : P.nu, sheet: null };
        res.hc = this.kernelsFrom(geo, this.amps(hc, vecs), c1, c2, P.D);
        if (this.sheetAvailable(P)) {
          await e.need('field', 'sheetp1', P);
          res.sheet = this.kernelsFrom(geo, this.amps(e.delta('sheetp1', P), vecs), c1, c2, P.D);
        }
        this.res = res;
      },
    });
    t.push({ label: 'Kernels: sweep', heavy: P.mode === 3, fn: () => this.sweep(P) });
    t.push({ label: 'Kernels: draw', fn: () => this.draw(P) });
    return t;
  }

  /** Sweep settings for P: {allowed, npts, kind}. */
  sweepPlan(P) {
    const d3 = P.mode === 3;
    return { allowed: !d3 || P.n <= 64, npts: d3 ? 6 : 10, kind: P.kx === 'nu' ? 'nu' : 'D' };
  }

  sweepKey(P) {
    const pl = this.sweepPlan(P);
    return ['kxsweep', pl.kind, pl.npts, P.D, P.nu, P.me, P.mx, P.hs, P.order];
  }

  /** First shell-crossing time of the applied order (2D: computed on demand; 3D: only when already available). */
  dscOf(P, order = P.order) {
    const e = this.app.eng;
    if (P.mode === 2) { try { return e.dsc(order); } catch (err) { return NaN; } }
    const v = e.peek(['dsc', order]);
    return v === undefined ? NaN : v;
  }

  /** The Hopf-Cole solve of the sweep (same source / method / max_exp as the engine's 'hc' field) and its density + effective nu. */
  hcSolve(P, d, nu) {
    const e = this.app.eng, s = e.sim;
    if ((P.hs || 'zel') === 'zel') s.hopf_cole(d, nu, P.me, P.mx);
    else { e.ensureWasm(P.order); s.hopf_cole_lpt(d, P.order, nu, P.me, P.mx, P.hs === 'lptT' && P.order > 2); }
    return { delta: s.hc_delta(), nuEff: s.hc_nu_eff() };
  }

  async sweep(P) {
    const app = this.app, e = app.eng, plan = this.sweepPlan(P);
    if (!plan.allowed || !e.sim || e.dim !== P.mode) return;
    const key = this.sweepKey(P);
    if (e.peek(key) !== undefined) return;
    const gen = app.runGen;
    const geo = this.geometry(P);
    if (!geo.ok) return;
    const { c1, c2 } = this.linearAmps(P, geo);
    const hs = geo.rows.slice(1);
    const vecs = hs.map((r) => r.vec);
    const dsc = this.dscOf(P);
    let xs;
    if (plan.kind === 'D') {
      const ref = Number.isFinite(dsc) && dsc > 0 ? dsc : P.D;
      let lo = 0.02 * ref;
      if (!(lo < 0.98 * P.D)) lo = P.D / 50;
      xs = logspace(lo, P.D, plan.npts);
    } else xs = logspace(1e-5, 1e-2, plan.npts);
    const sheetSweep = plan.kind === 'D' && P.mode === 2;
    if (sheetSweep) e.ensureWasm(P.order);
    const out = { kind: plan.kind, xs, nuEff: new Float64Array(xs.length), hc: hs.map(() => new Float64Array(xs.length)), sheet: sheetSweep ? hs.map(() => new Float64Array(xs.length)) : null };
    const kerRow = (field, d) => {
      const K = this.kernelsFrom({ ...geo, rows: [geo.rows[0], ...hs] }, [[0, 0], ...this.amps(field, vecs)], c1, c2, d);
      return K.slice(1).map((k) => (k.ok ? k.re : NaN));
    };
    for (let i = 0; i < xs.length; i++) {
      app.setStatus(`computing: Kernels sweep ${i + 1}/${xs.length} (${plan.kind === 'D' ? 'D' : 'ν'} = ${fmtNum(xs[i], 3)}) …`, 'busy');
      await tick();
      if (gen !== app.runGen) return;
      const d = plan.kind === 'D' ? xs[i] : P.D, nu = plan.kind === 'nu' ? xs[i] : P.nu;
      const sol = this.hcSolve(P, d, nu);
      out.nuEff[i] = sol.nuEff;
      kerRow(sol.delta, d).forEach((v, h) => { out.hc[h][i] = v; });
      if (sheetSweep) {
        const rho = e.sim.sheet_density_p1(d, Math.min(P.order, e.wOrder), e.n, 2);
        const dl = new Float32Array(rho.length);
        for (let q = 0; q < dl.length; q++) dl[q] = rho[q] - 1;
        kerRow(dl, d).forEach((v, h) => { out.sheet[h][i] = v; });
      }
    }
    if (gen !== app.runGen) return;
    e.seed(key, out);
  }

  // ---------------------------------------------------------------- drawing

  draw(P) {
    const app = this.app, e = app.eng;
    this.noIc.hidden = P.ic === 'w';
    this.main.hidden = P.ic !== 'w';
    if (P.ic !== 'w') { this.note.hidden = true; this.markStale(false); return; }
    if (!e.sim || e.dim !== P.mode || !this.res || this.res.P !== P) { this.markStale(false); return; }
    this.kxSel.set(P.kx);
    this.drawTable(P);
    this.drawSweep(P);
    this.updateReadouts(P, { dsc: this.dscOf(P) });
    this.markStale(false);
  }

  cellKernel(td, k, ref) {
    if (!k || !k.ok) { td.textContent = '—'; td.title = 'not available (zero or degenerate mode, or an amplitude of the linear field is zero)'; return; }
    el('span', null, td, fmtNum(k.re, 4));
    el('span', 'lab-kern-im', td, ` i${k.im < 0 ? '−' : '+'}${fmtNum(Math.abs(k.im), 2)}`);
    td.title = `Re = ${k.re.toPrecision(5)}, Im = ${k.im.toExponential(2)} (should be ≈ 0)` + (ref ? `\nratio to F₂^ν: ${(k.re / ref).toPrecision(4)}` : '');
  }

  drawTable(P) {
    const r = this.res, geo = r.geo, dim = geo.dim;
    const host = this.tw;
    host.textContent = '';
    const tb = el('table', 'lab-pdf-table lab-kern-table', host);
    const hr = el('tr', null, el('thead', null, tb));
    const cols = [
      ['harmonic', 'the mode: first-order propagator at k₁, then the second-order harmonics'],
      ['mode', 'integer mode vector in units of the fundamental 2π/L'],
      ['Hopf–Cole', `measured kernel of the Hopf–Cole density (ν = ${fmtNum(P.nu, 3)}, ν_eff = ${fmtNum(r.nuEff, 3)}, method ${P.me ? 'log-domain ×' + P.me : 'Fourier multiplier'}): c / (mult · D² · c₀ products)`],
      [`sheet P1 (${P.order}LPT)`, 'measured kernel of the P1 sheet density at the current LPT order'],
      ['F₂^ν', 'analytic viscous kernel (adhesion model) with the effective viscosity of the solver; for k₁: exp(−ν k₁² D)'],
      ['F₂^ZA', 'Zel’dovich kernel (ν → 0 limit of the viscous kernel); 1 for k₁'],
      ['F₂ grav', 'gravitational Einstein–de Sitter kernel F₂; 1 for k₁'],
    ];
    for (const [c, tip] of cols) { const th = el('th', null, hr, c); th.title = tip; }
    const body = el('tbody', null, tb);
    geo.rows.forEach((row, i) => {
      const tr = el('tr', null, body);
      const lab = el('td', 'lab-kern-name', tr);
      if (i > 0) { const sw = el('span', 'lab-pdf-swatch', lab); sw.style.background = HCOL[i - 1]; }
      el('span', null, lab, i === 0 ? 'k₁ (linear)' : row.label);
      if (row.bad) lab.title = row.bad;
      el('td', null, tr, vstr(row.vec, dim));
      const an = row.bad ? [NaN, NaN, NaN] : this.analytic(geo, i, r.nuEff, P.D);
      const tdH = el('td', null, tr), tdS = el('td', null, tr);
      this.cellKernel(tdH, r.hc[i], an[0]);
      if (r.sheet) this.cellKernel(tdS, r.sheet[i], an[0]); else { tdS.textContent = '—'; tdS.title = 'the tetrahedral sheet is not computed at this grid size without the GPU compute path'; }
      for (const v of an) el('td', null, tr, Number.isFinite(v) ? fmtNum(v, 4) : '—');
    });
    this.tTitle.textContent = `Kernels at D = ${fmtNum(P.D, 3)}, ν = ${fmtNum(P.nu, 3)}`;
    const notes = [];
    if (!r.sheet) notes.push('The sheet column is not shown: the tetrahedral sheet takes tens of seconds in WASM at this grid size (shown at 64³ or below, or with GPU compute).');
    const badRows = geo.rows.filter((q) => q.bad).map((q) => `${q.label} (${q.bad})`);
    if (badRows.length) notes.push(`Not separable for these mode vectors: ${badRows.join(', ')}.`);
    if (cabs(r.c1) < C0_MIN || cabs(r.c2) < C0_MIN) notes.push('One of the two waves has zero amplitude.');
    this.note.hidden = !notes.length;
    this.note.textContent = notes.join(' ');
  }

  drawSweep(P) {
    const e = this.app.eng, plan = this.sweepPlan(P), geo = this.res.geo;
    const plot = this.plot;
    const sw = plan.allowed ? e.peek(this.sweepKey(P)) : undefined;
    if (!plan.allowed || !sw) {
      plot.setSeries([]); plot.setMarkers([]); plot.setAxes({ xlog: false, ylog: false, xlabel: '', ylabel: '' }); plot.draw();
      this.pTitle.textContent = 'Kernel sweep';
      this.cap.textContent = !plan.allowed
        ? `The sweep is skipped in 3D above 64³ (every point is a full Hopf–Cole solve in WASM): the table above is the whole check. Use 64³ or fewer cells (the sweep then has ${plan.npts} points), or switch to 2D.`
        : 'The sweep was not computed (cancelled or not yet run).';
      return;
    }
    const isD = sw.kind === 'D', xs = sw.xs, xa = xs[0], xb = xs[xs.length - 1];
    const dense = logspace(xa, xb, 80);
    // effective viscosity at the dense points: log-x interpolation of the solver's nu_eff
    const nuAt = (x) => {
      let i = 0; while (i < xs.length - 2 && x > xs[i + 1]) i++;
      const t = Math.min(1, Math.max(0, Math.log(x / xs[i]) / Math.log(xs[i + 1] / xs[i])));
      const a = sw.nuEff[i], b = sw.nuEff[i + 1];
      return Number.isFinite(a) && Number.isFinite(b) ? a * Math.pow(b / a, t) : P.nu;
    };
    const series = [];
    const hs = geo.rows.slice(1);
    hs.forEach((row, h) => {
      const col = HCOL[h];
      if (row.bad) return;
      series.push({ x: dense, y: Float64Array.from(dense, (x) => this.f2(2, row.a, row.b, nuAt(x), isD ? x : P.D)), color: col, width: 1.4, dash: '6 3', opacity: 0.9 });
      series.push({ x: [xa, xb], y: [this.f2(1, row.a, row.b, P.nu, P.D), this.f2(1, row.a, row.b, P.nu, P.D)], color: col, width: 1.2, dash: '1.5 3', opacity: 0.9 });
      series.push({ x: [xa, xb], y: [this.f2(0, row.a, row.b, P.nu, P.D), this.f2(0, row.a, row.b, P.nu, P.D)], color: col, width: 0.9, opacity: 0.5 });
      if (sw.sheet) series.push({ x: xs, y: sw.sheet[h], color: col, points: true, hollow: true, line: false, radius: 3.6 });
      series.push({ x: xs, y: sw.hc[h], color: col, points: true, width: 1.5, label: row.label, radius: 3 });
    });
    plot.setAxes({
      xlog: true, ylog: false, xlabel: isD ? 'growth factor D' : 'viscosity ν', ylabel: 'kernel K = c / (mult · D² · c₀ products)',
      xlim: [xa, xb],
    });
    plot.setSeries(series);
    // y limits: the data (clipping outliers of the noisy small-D points to the analytic range), with headroom for the legend
    let lo = Infinity, hi = -Infinity;
    for (const q of series) if (!q.points) for (const v of q.y) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo < hi) { const sp = hi - lo; plot.setAxes({ ylim: [lo - 0.1 * sp, hi + 0.4 * sp] }); }
    const dsc = this.dscOf(P);
    const mk = [];
    if (isD && Number.isFinite(dsc) && dsc > xa && dsc < xb) mk.push({ x: dsc, label: 'D_sc', color: '#8a8f98' });
    // grid floor of the log-domain solver: a heat kernel narrower than a cell is not represented, nu_eff D < dx^2/4 (the Hopf-Cole points left of / below it are under-resolved)
    if (isD) mk.push({ x: 1 / (4 * P.n * P.n * this.res.nuEff), label: 'D_floor', color: '#c0504d' });
    else mk.push({ x: P.nu, label: 'ν', color: '#8a8f98' }, { x: 1 / (4 * P.n * P.n * P.D), label: 'ν_floor', color: '#c0504d' });
    plot.setMarkers(mk);
    plot.draw();
    this.pTitle.textContent = isD ? `Kernels versus D (ν = ${fmtNum(P.nu, 3)})` : `Kernels versus ν (D = ${fmtNum(P.D, 3)})`;
    this.cap.textContent = 'Filled points: Hopf–Cole density, re-solved at every sweep value. Dashed: analytic viscous F₂^ν (with the solver’s ν_eff). Fine dotted: F₂^ZA; thin solid: gravitational F₂. '
      + (sw.sheet ? `Hollow points: P1 sheet at ${P.order}LPT. ` : (isD ? '' : 'The sheet does not depend on ν. '))
      + `Colours: ${hs.map((q) => q.label).join(', ')}. ` + (isD ? 'Departures at the largest D are higher orders (∝ D² relative); left of the red D_floor = Δx²/(4ν_eff) the log-domain solver cannot resolve its heat kernel (and the sheet deposit noise is large compared with D²).'
        : 'Departures at the largest ν are viscous damping beyond second order; below the red ν_floor = Δx²/(4D) the log-domain solver cannot resolve its heat kernel.');
  }

  updateReadouts(P, { dsc } = {}) {
    if (P.ic !== 'w' || !this.res || this.res.P.mode !== P.mode) { this.readout.textContent = ''; return; }
    const geo = this.res.geo, kf = this.app.eng.kf || 2 * Math.PI;
    const th = Math.acos(Math.max(-1, Math.min(1, geo.cos))) * 180 / Math.PI;
    const tidal = 2 / 7 * (1 - geo.cos * geo.cos);
    const nk = (k) => this.res.nuEff * kf * kf * vdot(k, k) * P.D;
    const nuTxt = Math.abs(this.res.nuEff / P.nu - 1) > 0.01 ? `ν = ${fmtNum(P.nu, 3)}, ν_eff = ${fmtNum(this.res.nuEff, 3)}` : `ν = ${fmtNum(P.nu, 3)}`;
    const dscTxt = Number.isFinite(dsc) ? `D_sc(${P.order}LPT) = ${fmtNum(dsc, 3)}, D/D_sc = ${fmtNum(P.D / dsc, 3)}` : 'no shell crossing found (or not yet computed)';
    this.readout.textContent = `tidal term F₂ − F₂^ZA = 3/14[1−cos²θ] = ${fmtNum(tidal * 0.75, 4)} (θ = ${fmtNum(th, 4)}°)`
      + ` · ν k₁² D = ${fmtNum(nk(geo.k1), 3)}, ν k₂² D = ${fmtNum(nk(geo.k2), 3)} (${nuTxt}, D = ${fmtNum(P.D, 3)})`
      + ` · ${dscTxt}`
      + (Number.isFinite(dsc) && P.D > 0.5 * dsc ? ' · D is not small compared with D_sc: higher orders (∝ D² relative) contaminate the second-order comparison' : '');
  }

  destroy() { this.plot.destroy(); this.root.remove(); }
}
