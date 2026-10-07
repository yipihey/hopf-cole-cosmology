// Chapter widgets: small interactive panels embedded in the Quarto chapters.
// Each <div class="hcc-widget" data-widget="name"> is mounted by mountAll().

import { loadCore, kaxis, halfSpectrum, breakingTime, sampleProfile, PROFILES, randomField1D, linspaceBox, transpose2D, percentiles, fmap } from './hcc.js';
import { LinePlot, PALETTE } from './viz/plot.js';
import { slider, select, checkbox, readout, row, button } from './viz/ui.js';
import { FieldView } from './viz/fieldview.js';
import { renderColorbar } from './viz/colormaps.js';

const C = PALETTE;
const GRAY = '#999999';

function panel(parent, title, cls = '') {
  const p = document.createElement('div');
  p.className = 'hcc-panel ' + cls;
  if (title) { const h = document.createElement('div'); h.className = 'hcc-title'; h.innerHTML = title; p.appendChild(h); }
  parent.appendChild(p);
  return p;
}
function grid(parent, cols = 2) {
  const g = document.createElement('div');
  g.className = cols === 3 ? 'hcc-grid3' : 'hcc-grid2';
  parent.appendChild(g);
  return g;
}
function note(parent, html) {
  const n = document.createElement('p'); n.className = 'hcc-note'; n.innerHTML = html; parent.appendChild(n); return n;
}
function controls(parent) { const c = document.createElement('div'); c.className = 'hcc-controls'; parent.appendChild(c); return c; }
function profileOptions(keys) { return keys.map((k) => ({ value: k, label: PROFILES[k].label })); }
/** Debounced, coalescing scheduler: at most one update in flight, latest request wins. */
function scheduler(fn) {
  let pending = false, running = false;
  const run = () => {
    if (running) { pending = true; return; }
    running = true; pending = false;
    requestAnimationFrame(() => {
      try { fn(); } catch (e) { console.error(e); }
      running = false;
      if (pending) run();
    });
  };
  return run;
}
const wrapHalf = (x, L) => ((x + L / 2) % L + L) % L - L / 2; // into [-L/2, L/2)

// ---------------------------------------------------------------------------
// 1. Burgers: pseudo-spectral RK4 vs exact Hopf–Cole, with characteristics
// ---------------------------------------------------------------------------
function burgers1d(el, core) {
  const N = 512, L = 2;
  const x = linspaceBox(N, L, -L / 2);
  const dx = L / N;
  const k = kaxis(N, L);
  const ctl = controls(el);
  const prof = select(ctl, { label: 'initial u₀', options: profileOptions(['sech', 'sech2', 'sine', 'twoSine']), value: 'sech', onChange: () => update() });
  const nu = slider(ctl, { label: 'ν', min: 1e-4, max: 0.05, value: 0.005, log: true, onInput: () => update() });
  const t = slider(ctl, { label: 't', min: 0, max: 1.2, step: 0.005, value: 0, onInput: () => update() });
  const showHC = checkbox(ctl, { label: 'exact Hopf–Cole (dashed)', value: true, onChange: () => update() });
  const showChar = checkbox(ctl, { label: 'inviscid characteristics', value: false, onChange: () => update() });
  const dealias = checkbox(ctl, { label: '2/3 dealiasing', value: true, onChange: () => update() });
  const tstar = readout(ctl, { label: 't⋆' });
  const g = grid(el);
  const p1 = new LinePlot(panel(g, 'Real space: u(x, t)'), { width: 520, height: 320 });
  const p2 = new LinePlot(panel(g, 'Fourier space: |û<sub>k</sub>|'), { width: 520, height: 320 });
  note(el, 'Solid: pseudo-spectral integrating-factor RK4 (the notebook method). Dashed: exact Hopf–Cole solution. Dotted: the multi-valued inviscid solution x = x₀ + u₀(x₀) t. t⋆ = −1/min u₀′ is the breaking time.');
  const update = scheduler(() => {
    const u0 = sampleProfile(prof.get(), N, L);
    const ts = breakingTime(u0, dx);
    tstar.set(Number.isFinite(ts) ? ts.toFixed(3) : '∞');
    const tt = t.get();
    const uRK = core.burgers_rk4(u0, L, nu.get(), tt, dealias.get());
    const series = [{ x, y: u0, label: 'u₀', color: GRAY, width: 1 }, { x, y: uRK, label: 'RK4', color: C[0], width: 2 }];
    if (showHC.get()) {
      const hc = core.hopf_cole_1d(u0, L, nu.get(), tt);
      series.push({ x, y: hc.u(), label: 'Hopf–Cole', color: C[1], dash: '6 4', width: 2 });
      hc.free();
    }
    if (showChar.get() && tt > 0) {
      const inv = core.inviscid_1d(u0, L, tt);
      const xc = inv.x_char();
      const xw = new Float64Array(N); for (let i = 0; i < N; i++) xw[i] = wrapHalf(xc[i] - L / 2, L);
      // draw as points to allow the fold
      series.push({ x: xw, y: u0, label: 'characteristics', color: C[2], line: false, points: true, radius: 1.2, opacity: 0.8 });
      inv.free();
    }
    p1.setAxes({ xlabel: 'x', ylabel: 'u', xlim: [-L / 2, L / 2] });
    p1.setSeries(series); p1.setMarkers([]); p1.draw();
    const s0 = halfSpectrum(core.spectrum_1d(u0)), s1 = halfSpectrum(core.spectrum_1d(uRK));
    const kk = k.subarray(1), ref = fmap(kk, (q) => s1[1] * kk[0] / q);
    p2.setAxes({ xlabel: 'k', ylabel: '|û_k|', xlog: true, ylog: true, ylim: [1e-8, 1] });
    p2.setSeries([{ x: kk, y: s0.subarray(1), label: 't = 0', color: GRAY, width: 1 }, { x: kk, y: s1.subarray(1), label: `t = ${tt.toFixed(2)}`, color: C[0], width: 2 },
      { x: kk, y: ref, label: 'k⁻¹ (sawtooth)', color: C[3], dash: '2 3', width: 1 }]);
    p2.setMarkers([{ x: (2 * Math.PI / L) * (N / 3), label: '2/3 cut', color: GRAY }]);
    p2.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 2. Heat equation: kernel and multiplier side by side
// ---------------------------------------------------------------------------
function heat1d(el, core) {
  const N = 512, L = 2;
  const x = linspaceBox(N, L, -L / 2);
  const k = kaxis(N, L);
  const ics = {
    step: { label: 'square pulse', f: (xx) => (Math.abs(xx) < 0.25 ? 1 : 0) },
    gauss: { label: 'narrow Gaussian', f: (xx) => Math.exp(-(xx * xx) / (2 * 0.02 ** 2)) },
    sines: { label: 'three sines', f: (xx) => 0.5 + 0.3 * Math.sin(2 * Math.PI * xx / L) + 0.2 * Math.sin(10 * Math.PI * xx / L) + 0.15 * Math.sin(40 * Math.PI * xx / L) },
    noise: { label: 'white noise', f: null },
  };
  const ctl = controls(el);
  const ic = select(ctl, { label: 'ψ₀', options: Object.entries(ics).map(([v, o]) => ({ value: v, label: o.label })), value: 'step', onChange: () => update() });
  const nu = slider(ctl, { label: 'ν', min: 1e-4, max: 0.05, value: 0.002, log: true, onInput: () => update() });
  const t = slider(ctl, { label: 't', min: 0, max: 10, step: 0.01, value: 0.5, onInput: () => update() });
  const ldiff = readout(ctl, { label: '√(νt)' });
  const g = grid(el);
  const p1 = new LinePlot(panel(g, 'Real space: ψ(x,t) = (G<sub>t</sub> ∗ ψ₀)(x)'), { width: 520, height: 320 });
  const p2 = new LinePlot(panel(g, 'Fourier space: ψ̂<sub>k</sub>(t) = ψ̂<sub>k</sub>(0) e<sup>−νk²t</sup>'), { width: 520, height: 320 });
  note(el, 'Left: the solution (blue), the initial condition (grey) and the heat kernel G<sub>t</sub> (dashed, scaled to the plot). Right: Fourier amplitudes before and after, with the multiplier e<sup>−νk²t</sup> (dashed, scaled). The kernel width √(2νt) and the multiplier width 1/√(νt) are inverse to each other.');
  let noise = null;
  const update = scheduler(() => {
    const key = ic.get();
    let psi0;
    if (key === 'noise') {
      if (!noise) noise = randomField1D(N, L, 12345, 0, 0);
      psi0 = fmap(noise, (v) => 0.5 + 0.3 * v); psi0 = Float64Array.from(psi0);
    } else { psi0 = new Float64Array(N); for (let i = 0; i < N; i++) psi0[i] = ics[key].f(x[i]); }
    const tt = t.get(), nn = nu.get();
    ldiff.set(Math.sqrt(nn * tt).toFixed(3));
    const psi = core.heat_1d(psi0, L, nn, tt);
    const s = 4 * nn * tt;
    const peak = Math.max(...psi0) || 1;
    const G = new Float64Array(N);
    if (tt > 0) for (let i = 0; i < N; i++) G[i] = Math.exp(-(x[i] * x[i]) / s) * peak;
    p1.setAxes({ xlabel: 'x', ylabel: 'ψ', xlim: [-L / 2, L / 2] });
    p1.setSeries([{ x, y: psi0, label: 'ψ₀', color: GRAY, width: 1 }, { x, y: psi, label: `ψ(t = ${tt.toFixed(2)})`, color: C[0], width: 2 },
      { x, y: G, label: 'kernel (scaled)', color: C[1], dash: '6 4', width: 1.5 }]);
    p1.draw();
    const a0 = halfSpectrum(core.spectrum_1d(psi0)), a1 = halfSpectrum(core.spectrum_1d(psi));
    const kk = k.subarray(1);
    const mult = fmap(kk, (q) => a0[1] * Math.exp(-nn * q * q * tt));
    p2.setAxes({ xlabel: 'k', ylabel: '|ψ̂_k|', xlog: true, ylog: true, ylim: [1e-9, 1] });
    p2.setSeries([{ x: kk, y: a0.subarray(1), label: 't = 0', color: GRAY, width: 1 }, { x: kk, y: a1.subarray(1), label: `t = ${tt.toFixed(2)}`, color: C[0], width: 2 },
      { x: kk, y: mult, label: 'multiplier (scaled)', color: C[1], dash: '6 4', width: 1.5 }]);
    p2.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 3. The Hopf–Cole chain: u → Φ → ψ → ψ̂ e^{-νk²t} → ψ → Φ → u
// ---------------------------------------------------------------------------
function hopfcole1d(el, core) {
  const N = 512, L = 2;
  const x = linspaceBox(N, L, -L / 2);
  const k = kaxis(N, L);
  const ctl = controls(el);
  const prof = select(ctl, { label: 'u₀', options: profileOptions(['sech', 'sech2', 'sine', 'twoSine']), value: 'sech', onChange: () => update() });
  const nu = slider(ctl, { label: 'ν', min: 1e-3, max: 0.1, value: 0.009, log: true, onInput: () => update() });
  const t = slider(ctl, { label: 't', min: 0, max: 1.0, step: 0.005, value: 0.3, onInput: () => update() });
  const rangeOut = readout(ctl, { label: 'ΔΦ/2ν' });
  const g = grid(el);
  const pU = new LinePlot(panel(g, '① u(x): the velocity'), { width: 520, height: 280 });
  const pPhi = new LinePlot(panel(g, '② Φ(x) = ∫u dx: the potential'), { width: 520, height: 280 });
  const pPsi = new LinePlot(panel(g, '③ ψ = e<sup>−Φ/2ν</sup> (log scale)'), { width: 520, height: 280 });
  const pHat = new LinePlot(panel(g, '④ |ψ̂<sub>k</sub>| ← the only dynamics is e<sup>−νk²t</sup>'), { width: 520, height: 280 });
  note(el, 'Grey curves are the initial state, coloured curves the state at time t. Panels ②–④ are the exact solution path: Φ₀ → ψ₀ → multiply by e<sup>−νk²t</sup> → ψ(t) → Φ(t) = −2ν ln ψ → u = Φ<sub>x</sub>. ΔΦ/2ν is the number of e-folds spanned by ψ: in double precision the Fourier method fails beyond about 30.');
  const update = scheduler(() => {
    const u0 = sampleProfile(prof.get(), N, L);
    const tt = t.get(), nn = nu.get();
    const hc = core.hopf_cole_1d(u0, L, nn, tt);
    const u = hc.u(), phi = hc.phi(), phi0 = hc.phi0(), psi = hc.psi(), psi0 = hc.psi0();
    const ph0 = hc.psihat0_abs(), ph1 = hc.psihat_abs();
    hc.free();
    let pmin = Infinity, pmax = -Infinity; for (const v of phi0) { if (v < pmin) pmin = v; if (v > pmax) pmax = v; }
    rangeOut.set(((pmax - pmin) / (2 * nn)).toFixed(1));
    pU.setAxes({ xlabel: 'x', ylabel: 'u', xlim: [-L / 2, L / 2] });
    pU.setSeries([{ x, y: u0, label: 'u₀', color: GRAY, width: 1 }, { x, y: u, label: `u(t = ${tt.toFixed(2)})`, color: C[0], width: 2 }]); pU.draw();
    pPhi.setAxes({ xlabel: 'x', ylabel: 'Φ', xlim: [-L / 2, L / 2] });
    pPhi.setSeries([{ x, y: phi0, label: 'Φ₀', color: GRAY, width: 1 }, { x, y: phi, label: 'Φ(t)', color: C[1], width: 2 }]); pPhi.draw();
    const psi0n = fmap(psi0, (v) => Math.max(v, 1e-300)), psin = fmap(psi, (v) => Math.max(v, 1e-300));
    pPsi.setAxes({ xlabel: 'x', ylabel: 'ψ', xlim: [-L / 2, L / 2], ylog: true });
    pPsi.setSeries([{ x, y: psi0n, label: 'ψ₀', color: GRAY, width: 1 }, { x, y: psin, label: 'ψ(t)', color: C[2], width: 2 }]); pPsi.draw();
    const kk = k.subarray(1);
    const h0 = fmap(halfSpectrum(ph0).subarray(1), (v) => v / N + 1e-300), h1 = fmap(halfSpectrum(ph1).subarray(1), (v) => v / N + 1e-300);
    const mult = fmap(kk, (q) => h0[0] * Math.exp(-nn * q * q * tt));
    pHat.setAxes({ xlabel: 'k', ylabel: '|ψ̂_k|', xlog: true, ylog: true, ylim: [1e-12, 1] });
    pHat.setSeries([{ x: kk, y: h0, label: 't = 0', color: GRAY, width: 1 }, { x: kk, y: h1, label: 't', color: C[2], width: 2 }, { x: kk, y: mult, label: 'e^{−νk²t} (scaled)', color: C[1], dash: '6 4', width: 1.5 }]);
    pHat.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 4. Mode coupling: a single sine mode feeds its harmonics
// ---------------------------------------------------------------------------
function modes1d(el, core) {
  const N = 512, L = 2 * Math.PI;
  const x = linspaceBox(N, L, -L / 2);
  const k = kaxis(N, L);
  const ctl = controls(el);
  const prof = select(ctl, { label: 'u₀', options: [{ value: 'sine', label: 'sin x' }, { value: 'twoSine', label: 'sin x + ½ sin 3x' }], value: 'sine', onChange: () => update() });
  const nu = slider(ctl, { label: 'ν', min: 1e-3, max: 0.1, value: 0.01, log: true, onInput: () => update() });
  const t = slider(ctl, { label: 't', min: 0, max: 3, step: 0.01, value: 0.5, onInput: () => update() });
  const tstar = readout(ctl, { label: 't⋆' });
  const g = grid(el);
  const p1 = new LinePlot(panel(g, 'u(x, t) exact (Hopf–Cole)'), { width: 520, height: 300 });
  const p2 = new LinePlot(panel(g, 'Harmonics |û<sub>n</sub>| vs n'), { width: 520, height: 300 });
  const p3 = new LinePlot(panel(el, 'Growth of the first harmonics in time'), { width: 1060, height: 260 });
  note(el, 'The first few harmonics grow as t<sup>n−1</sup> at early times (n-th order in perturbation theory: n−1 mode couplings), approach the k<sup>−1</sup> sawtooth spectrum near the breaking time t⋆ = 1, and are then cut off by viscosity at k ≈ Δu/ν. Grey: the “linear” solution, in which mode 1 only decays and no harmonic is ever generated.');
  const update = scheduler(() => {
    const u0 = sampleProfile(prof.get(), N, L);
    tstar.set(breakingTime(u0, L / N).toFixed(3));
    const tt = t.get(), nn = nu.get();
    const hc = core.hopf_cole_1d(u0, L, nn, tt);
    const u = hc.u(), uh = halfSpectrum(hc.uhat_abs());
    hc.free();
    const ulin = fmap(u0, (v) => v * Math.exp(-nn * tt)); // single mode decays: exp(-ν k² t) with k = 1
    p1.setAxes({ xlabel: 'x', ylabel: 'u', xlim: [-L / 2, L / 2] });
    p1.setSeries([{ x, y: u0, label: 'u₀', color: GRAY, width: 1 }, { x, y: ulin, label: 'linear (mode 1 decays)', color: GRAY, dash: '4 3', width: 1 }, { x, y: u, label: `t = ${tt.toFixed(2)}`, color: C[0], width: 2 }]);
    p1.draw();
    const nmax = 48;
    const nn_ = new Float64Array(nmax), amp = new Float64Array(nmax), ref = new Float64Array(nmax), lin = new Float64Array(nmax);
    for (let n = 1; n <= nmax; n++) { nn_[n - 1] = n; amp[n - 1] = 2 * uh[n] + 1e-300; ref[n - 1] = 2 * uh[1] / n + 1e-300; lin[n - 1] = n === 1 ? Math.exp(-nn * tt) : 1e-300; }
    p2.setAxes({ xlabel: 'harmonic n', ylabel: '|û_n|', xlog: true, ylog: true, ylim: [1e-6, 2] });
    p2.setSeries([{ x: nn_, y: amp, label: 'exact', color: C[0], width: 2, points: true }, { x: nn_, y: ref, label: 'n⁻¹ sawtooth', color: C[3], dash: '2 3', width: 1 }, { x: nn_, y: lin, label: 'linear', color: GRAY, line: false, points: true }]);
    p2.setMarkers(nn > 0 ? [{ x: Math.min(nmax, 2 / nn), label: 'k ≈ Δu/ν', color: GRAY }] : []);
    p2.draw();
    // time series of harmonics 1..5
    const nt = 60, tsr = new Float64Array(nt), hs = [1, 2, 3, 4, 5].map(() => new Float64Array(nt));
    for (let i = 0; i < nt; i++) {
      const ti = (i + 1) * 3 / nt; tsr[i] = ti;
      const h = core.hopf_cole_1d(u0, L, nn, ti); const a = halfSpectrum(h.uhat_abs()); h.free();
      for (let m = 0; m < 5; m++) hs[m][i] = 2 * a[m + 1] + 1e-300;
    }
    p3.setAxes({ xlabel: 't', ylabel: '|û_n|', xlog: true, ylog: true, ylim: [1e-5, 2] });
    p3.setSeries(hs.map((y, m) => ({ x: tsr, y, label: `n = ${m + 1}`, color: C[m], width: 1.5 })).concat([{ x: tsr, y: fmap(tsr, (q) => 0.5 * q), label: '∝ t', color: GRAY, dash: '2 3', width: 1 }, { x: tsr, y: fmap(tsr, (q) => 0.375 * q * q), label: '∝ t²', color: GRAY, dash: '6 3', width: 1 }]));
    p3.setMarkers([{ x: tt, label: 't', color: C[0] }, { x: 1, label: 't⋆', color: GRAY }]);
    p3.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 5. Hopf–Lax: lower envelope of parabolas, inverse map, density
// ---------------------------------------------------------------------------
function hopflax1d(el, core) {
  const N = 512, L = 2;
  const x = linspaceBox(N, L, -L / 2);
  const ctl = controls(el);
  const prof = select(ctl, { label: 'u₀', options: profileOptions(['sine', 'twoSine', 'sech']).concat([{ value: 'random', label: 'random field' }]), value: 'sine', onChange: () => update() });
  const t = slider(ctl, { label: 't', min: 0.01, max: 1.5, step: 0.005, value: 0.2, onInput: () => update() });
  const xs = slider(ctl, { label: 'x', min: -L / 2, max: L / 2, step: 0.002, value: 0.15, onInput: () => update() });
  const tstar = readout(ctl, { label: 't⋆' });
  const g = grid(el);
  const p1 = new LinePlot(panel(g, 'Φ₀(y) and the touching parabola Φ(x,t) − (x−y)²/2t'), { width: 520, height: 320 });
  const p2 = new LinePlot(panel(g, 'u(x,t): entropy solution vs characteristics'), { width: 520, height: 320 });
  const g2 = grid(el);
  const p3 = new LinePlot(panel(g2, 'Inverse Lagrangian map y⋆(x)'), { width: 520, height: 260 });
  const p4 = new LinePlot(panel(g2, 'Density ρ/ρ̄ = dy⋆/dx'), { width: 520, height: 260 });
  note(el, 'Φ(x,t) = min<sub>y</sub>[Φ₀(y) + (x−y)²/2t]: the parabola centred on x (orange) slides up until it touches Φ₀ (blue) at y⋆. After the breaking time it can touch at two points for the same x: that x is the shock, y⋆(x) jumps there, and the density has a δ-function carrying the mass between the two touching points.');
  const update = scheduler(() => {
    const u0 = prof.get() === 'random' ? randomField1D(N, L, 7, -1, 0.05) : sampleProfile(prof.get(), N, L);
    const tt = t.get(), xx = xs.get();
    tstar.set(breakingTime(u0, L / N).toFixed(3));
    const inv = core.inviscid_1d(u0, L, tt);
    const phi = inv.phi(), ystar = inv.x0_star(), rho = inv.rho(), u = inv.u(), xc = inv.x_char();
    inv.free();
    const hc0 = core.hopf_cole_1d(u0, L, 0.01, 0); const phi0 = hc0.phi0(); hc0.free();
    // parabola for the chosen x
    const ix = Math.round((xx + L / 2) / L * N) % N;
    const phx = phi[ix];
    const par = new Float64Array(N);
    for (let i = 0; i < N; i++) { const d = wrapHalf(x[i] - xx, L); par[i] = phx - d * d / (2 * tt); }
    const ys = wrapHalf(ystar[ix] - L / 2, L);
    p1.setAxes({ xlabel: 'y', ylabel: 'Φ', xlim: [-L / 2, L / 2] });
    p1.setSeries([{ x, y: phi0, label: 'Φ₀(y)', color: C[0], width: 2 }, { x, y: phi, label: 'Φ(y, t) (envelope)', color: GRAY, width: 1 },
      { x, y: par, label: 'parabola at x', color: C[1], width: 1.5 }, { x: [ys], y: [phi0[Math.round((ys + L / 2) / L * N) % N]], label: 'y⋆', color: C[1], line: false, points: true, radius: 4 }]);
    p1.setMarkers([{ x: xx, label: 'x', color: C[1] }]); p1.draw();
    const xw = new Float64Array(N); for (let i = 0; i < N; i++) xw[i] = wrapHalf(xc[i] - L / 2, L);
    p2.setAxes({ xlabel: 'x', ylabel: 'u', xlim: [-L / 2, L / 2] });
    p2.setSeries([{ x: xw, y: u0, label: 'characteristics', color: C[2], line: false, points: true, radius: 1.2, opacity: 0.6 }, { x, y: u, label: 'u(x,t)', color: C[0], width: 2 }]);
    p2.setMarkers([{ x: xx, label: 'x', color: C[1] }]); p2.draw();
    const yw = fmap(ystar, (v) => wrapHalf(v - L / 2, L));
    p3.setAxes({ xlabel: 'x', ylabel: 'y⋆', xlim: [-L / 2, L / 2] });
    p3.setSeries([{ x, y: yw, label: 'y⋆(x)', color: C[0], width: 2 }, { x, y: x, label: 'identity', color: GRAY, dash: '4 3', width: 1 }]);
    p3.setMarkers([{ x: xx, label: 'x', color: C[1] }]); p3.draw();
    p4.setAxes({ xlabel: 'x', ylabel: 'ρ/ρ̄', xlim: [-L / 2, L / 2], ylog: true, ylim: [0.05, 200] });
    p4.setSeries([{ x, y: fmap(rho, (v) => Math.max(v, 0.05)), label: 'ρ/ρ̄', color: C[0], width: 2 }]); p4.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 6. One-dimensional cosmology: sheet vs adhesion
// ---------------------------------------------------------------------------
function zeldovich1d(el, core) {
  const N = 512, L = 1;
  const q = linspaceBox(N, L, 0);
  const ctl = controls(el);
  const ic = select(ctl, { label: 'δ₀', options: [{ value: 'wave', label: 'single wave −cos(2πx)' }, { value: 'two', label: 'two waves' }, { value: 'random', label: 'random field (R = 0.03)' }], value: 'wave', onChange: () => { needU = true; update(); } });
  const D = slider(ctl, { label: 'D', min: 0, max: 3, step: 0.005, value: 0.5, onInput: () => update() });
  const nu = slider(ctl, { label: 'ν', min: 1e-6, max: 1e-2, value: 1e-3, log: true, onInput: () => update() });
  const seedBtn = button(ctl, { label: 'new seed', onClick: () => { seed++; needU = true; update(); } });
  const dsc = readout(ctl, { label: 'D_sc' });
  const shockW = readout(ctl, { label: 'shock width ν/Δu (cells)' });
  const p1 = new LinePlot(panel(el, 'Phase space (x, u): the sheet folds; Burgers forms a shock'), { width: 1060, height: 300 });
  const p2 = new LinePlot(panel(el, 'Eulerian density ρ/ρ̄'), { width: 1060, height: 300 });
  note(el, 'Points: the Lagrangian sheet q ↦ (q + D u₀(q), u₀(q)), a multi-valued curve after shell crossing. Blue: the adhesion-model velocity at viscosity ν (Hopf–Cole, log-domain kernel). Densities: the multi-stream sheet sum Σ 1/|dx/dq| (points), the adhesion density 1 − D ∂<sub>x</sub>u (blue) and the inviscid Hopf–Lax density dy⋆/dx (grey) whose shocks are δ-functions. In 1D the Zel\'dovich solution is exact up to D<sub>sc</sub>. A Burgers shock has width ν/Δu: keep it above a grid cell or the density, a derivative of u, becomes noisy.');
  let seed = 3, needU = true, u0 = null;
  const update = scheduler(() => {
    if (needU) {
      const d0 = new Float64Array(N);
      if (ic.get() === 'wave') for (let i = 0; i < N; i++) d0[i] = -Math.cos(2 * Math.PI * q[i]);
      else if (ic.get() === 'two') for (let i = 0; i < N; i++) d0[i] = -Math.cos(2 * Math.PI * q[i]) + 0.7 * Math.cos(6 * Math.PI * q[i] + 1);
      else { const r = randomField1D(N, L, seed, -1, 0.03); for (let i = 0; i < N; i++) d0[i] = r[i]; }
      u0 = core.zeldovich_velocity_1d(d0, L);
      needU = false;
      dsc.set(breakingTime(u0, L / N).toFixed(3));
    }
    const dd = D.get(), nn = nu.get();
    { let lo = Infinity, hi = -Infinity; for (const v of u0) { if (v < lo) lo = v; if (v > hi) hi = v; } shockW.set((nn / (hi - lo) / (L / N)).toFixed(2)); }
    const xpos = new Float64Array(N); for (let i = 0; i < N; i++) xpos[i] = q[i] + dd * u0[i];
    const xw = fmap(xpos, (v) => ((v % L) + L) % L);
    const rhoSheet = core.sheet_density_1d(xpos, L, N);
    const hc = core.hopf_cole_1d(u0, L, nn, dd); const uhc = hc.u(); hc.free();
    const dxq = L / N;
    const rhoHC = new Float32Array(N);
    for (let i = 0; i < N; i++) { // 4th-order central difference (local, no Gibbs ringing at shocks)
      const du = (8 * (uhc[(i + 1) % N] - uhc[(i + N - 1) % N]) - (uhc[(i + 2) % N] - uhc[(i + N - 2) % N])) / (12 * dxq);
      rhoHC[i] = Math.max(1 - dd * du, 1e-3);
    }
    let rhoHL = null, uHL = null;
    if (dd > 0) { const inv = core.inviscid_1d(u0, L, dd); rhoHL = inv.rho(); uHL = inv.u(); inv.free(); }
    const series = [{ x: xw, y: u0, label: 'sheet (Lagrangian points)', color: C[1], line: false, points: true, radius: 1.3, opacity: 0.7 },
      { x: q, y: uhc, label: `adhesion, ν = ${nn.toExponential(0)}`, color: C[0], width: 2 }];
    if (uHL) series.push({ x: q, y: uHL, label: 'Hopf–Lax (ν → 0)', color: GRAY, dash: '4 3', width: 1 });
    p1.setAxes({ xlabel: 'x', ylabel: 'u = dx/dD', xlim: [0, L] });
    p1.setSeries(series); p1.draw();
    const s2 = [{ x: q, y: fmap(rhoSheet, (v) => Math.max(v, 1e-2)), label: 'sheet (multi-stream)', color: C[1], line: false, points: true, radius: 1.3, opacity: 0.8 },
      { x: q, y: rhoHC, label: 'adhesion (Hopf–Cole)', color: C[0], width: 2 }];
    if (rhoHL) s2.push({ x: q, y: fmap(rhoHL, (v) => Math.max(v, 1e-2)), label: 'Hopf–Lax', color: GRAY, dash: '4 3', width: 1 });
    p2.setAxes({ xlabel: 'x', ylabel: 'ρ/ρ̄', xlim: [0, L], ylog: true, ylim: [0.03, 300] });
    p2.setSeries(s2); p2.draw();
  });
  update();
}

// ---------------------------------------------------------------------------
// 7. LPT order by order (2D sheet densities and sources)
// ---------------------------------------------------------------------------
async function lptorders(el, core) {
  const N = 128;
  const ctl = controls(el);
  const R = slider(ctl, { label: 'smoothing R', min: 0.01, max: 0.08, value: 0.03, log: true, onInput: () => { needIC = true; update(); } });
  const D = slider(ctl, { label: 'D', min: 0.05, max: 1.5, step: 0.01, value: 0.5, onInput: () => update() });
  const seedBtn = button(ctl, { label: 'new seed', onClick: () => { seed++; needIC = true; update(); } });
  const dscOut = readout(ctl, { label: 'D_sc (orders 1–4)' });
  const status = readout(ctl, { label: '' });
  const g1 = document.createElement('div'); g1.className = 'hcc-grid2'; el.appendChild(g1);
  const g2 = document.createElement('div'); g2.className = 'hcc-grid2'; el.appendChild(g2);
  const views = [], srcViews = [];
  for (let n = 1; n <= 4; n++) {
    const p = panel(g1, `${n}LPT sheet density (log₁₀ ρ/ρ̄)`);
    const c = document.createElement('canvas'); c.style.cssText = 'width:100%;aspect-ratio:1'; p.appendChild(c);
    const v = new FieldView(c, { cmap: 'magma' }); await v.init(); views.push(v);
    const p2 = panel(g2, n === 1 ? 'source −δ₀ = ∇·Ψ⁽¹⁾' : `source ∇·Ψ⁽${n}⁾`);
    const c2 = document.createElement('canvas'); c2.style.cssText = 'width:100%;aspect-ratio:1'; p2.appendChild(c2);
    const v2 = new FieldView(c2, { cmap: 'rdbu' }); await v2.init(); srcViews.push(v2);
  }
  note(el, 'Top: Eulerian density from the phase-space sheet (triangles, point-sampled) at the same D for increasing LPT order; colour scale shared, log₁₀(ρ/ρ̄) from −1 to 1.5. Bottom: the longitudinal source ∇·Ψ⁽ⁿ⁾ of each order (symmetric colour range per panel). Beyond D<sub>sc</sub> the series no longer converges: watch orders 3 and 4 blow up in the densest regions.');
  let seed = 11, needIC = true, sim = null;
  const update = scheduler(() => {
    if (needIC) {
      if (sim) sim.free();
      sim = new core.CosmoSim(2, N, 1.0);
      sim.set_ic_gaussian(0, -1.0, 0, R.get(), seed, 1.0);
      sim.build_lpt(4);
      const ds = [1, 2, 3, 4].map((o) => sim.shell_crossing(o));
      dscOut.set(ds.map((v) => (Number.isFinite(v) ? v.toFixed(2) : '∞')).join(', '));
      for (let n = 1; n <= 4; n++) {
        const s = transpose2D(sim.lpt_div(n), N);
        const [lo, hi] = percentiles(s, 1, 99);
        const m = Math.max(Math.abs(lo), Math.abs(hi));
        srcViews[n - 1].setField(s, N, N); srcViews[n - 1].setRange(-m, m, { symmetric: true }); srcViews[n - 1].draw();
      }
      needIC = false;
    }
    const dd = D.get();
    const t0 = performance.now();
    for (let n = 1; n <= 4; n++) {
      const rho = transpose2D(sim.sheet_density(dd, n, N, 2), N);
      views[n - 1].setField(rho, N, N); views[n - 1].setRange(0.1, 30, { log: true }); views[n - 1].draw();
    }
    status.set(`${(performance.now() - t0).toFixed(0)} ms`);
  });
  update();
}

const WIDGETS = { burgers1d, heat1d, hopfcole1d, modes1d, hopflax1d, zeldovich1d, lptorders };

async function mountAll() {
  const els = document.querySelectorAll('.hcc-widget:not([data-mounted])');
  if (!els.length) return;
  let core;
  try { core = await loadCore(); } catch (e) {
    console.error(e);
    els.forEach((el) => { el.innerHTML = '<p class="hcc-note">The interactive panel could not load the WebAssembly module (' + e.message + ').</p>'; });
    return;
  }
  for (const el of els) {
    el.dataset.mounted = '1';
    const name = el.dataset.widget;
    const fn = WIDGETS[name];
    if (!fn) { el.innerHTML = `<p class="hcc-note">Unknown widget “${name}”.</p>`; continue; }
    el.classList.add('hcc-widget-mounted');
    try { await fn(el, core, el.dataset.opts ? JSON.parse(el.dataset.opts) : {}); }
    catch (e) { console.error(e); el.insertAdjacentHTML('beforeend', `<p class="hcc-note">Widget error: ${e.message}</p>`); }
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll); else mountAll();
