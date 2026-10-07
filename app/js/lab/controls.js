// controls.js - the left column of collapsible control groups.

import { slider, checkbox, button } from '../viz/ui.js';
import { el, sel, numIn, fmtNum } from './dom.js';
import { GRID_2D, GRID_3D, OMEGA_M } from './state.js';

const gridOptions = (mode) => (mode === 3 ? GRID_3D : GRID_2D).map((n) => [n, mode === 3 ? `${n}³` : `${n}²`]);
const methodOptions = (mode) => (mode === 3
  ? [[0, 'Fourier multiplier'], [1, 'log-domain ×1']]
  : [[0, 'Fourier multiplier'], [1, 'log-domain ×1'], [2, 'log-domain ×2'], [4, 'log-domain ×4']]);

function group(parent, title, open = true) {
  const d = el('details', 'lab-group', parent);
  d.open = open;
  el('summary', null, d, title);
  return el('div', 'lab-gbody', d);
}

export function buildControls(app, host) {
  const S = app.S;
  const C = {};
  const set = (key, level) => (v) => app.setParam(key, v, level);

  // ---------------------------------------------------------------- mode
  const gm = group(host, 'Mode');
  const modeRow = el('div', 'hcc-row lab-moderow', gm);
  C.modeRadios = [2, 3].map((m) => {
    const lab = el('label', 'hcc-ctl hcc-check lab-radio', modeRow);
    const r = el('input', null, lab); r.type = 'radio'; r.name = 'lab-mode'; r.value = m; r.checked = S.mode === m;
    el('span', 'hcc-label', lab, m === 2 ? '2D (fast, interactive)' : '3D (quantitative)');
    r.addEventListener('change', () => { if (r.checked) app.setMode(m); });
    return r;
  });
  C.n = sel(gm, { label: 'grid N', options: gridOptions(S.mode), value: S.n, onChange: (v) => app.setParam('n', Number(v), 'ic') });
  C.gc = checkbox(gm, { label: 'GPU compute', value: S.gc && app.gpuCompute, onChange: (v) => app.setGpuCompute(v) });
  C.gc.el.title = 'WebGPU compute path for the 3D mode: LPT positions, CIC deposit, Hopf–Cole log-sum-exp, FFT spectra and Fourier maps on the GPU (single precision). Off: WASM (double precision, seconds at 128³).';
  C.gcNote = el('p', 'lab-note-line', gm, 'GPU path: f32 arithmetic; CIC uses 18-bit fixed-point atomics (a cell overflows at ρ/ρ̄ ≥ 16384). LPT terms are uploaded once per build, so the D slider only recomputes the fields.');
  C.memWarn = el('p', 'lab-warn', gm, '128³ with 4LPT needs about 1 GB of memory and 30–60 s per run. Lower orders are cheaper.');

  // ---------------------------------------------------------------- initial conditions
  const gi = group(host, 'Initial conditions');
  C.ic = sel(gi, { label: 'type', options: [['g', 'Gaussian random field'], ['w', 'two plane waves'], ['p', 'single peak']], value: S.ic, onChange: (v) => { app.setParam('ic', v, 'ic'); C.syncVisibility(); } });

  // Gaussian field block
  C.gBlock = el('div', 'lab-block', gi);
  C.shape = sel(C.gBlock, { label: 'P(k)', options: [['pl', 'power law kⁿ'], ['bbks', 'BBKS (CDM-like)'], ['eh', 'Eisenstein–Hu no-wiggle']], value: S.shape,
    onChange: (v) => { app.setParam('shape', v, 'ic'); C.syncVisibility(); } });
  C.pn = slider(C.gBlock, { label: 'slope n', min: -2.5, max: 1, step: 0.05, value: S.pn, onInput: set('pn', 'ic'), format: (v) => v.toFixed(2) });
  C.ns = slider(C.gBlock, { label: 'tilt n_s', min: -1, max: 2, step: 0.05, value: S.ns, onInput: set('ns', 'ic'), format: (v) => v.toFixed(2) });
  C.gm = slider(C.gBlock, { label: 'Γ [1/L]', min: 2, max: 200, value: S.gm, log: true, onInput: set('gm', 'ic'), format: (v) => v.toFixed(1) });
  const seedRow = el('div', 'hcc-row', C.gBlock);
  C.seed = numIn(seedRow, { label: 'seed', value: S.seed, min: 0, max: 4294967295, step: 1, width: '6rem', onChange: set('seed', 'ic') });
  button(seedRow, { label: 'new seed', onClick: () => { const v = 1 + Math.floor(Math.random() * 999999); C.seed.set(v); app.setParam('seed', v, 'ic'); } });

  // plane waves block
  C.wBlock = el('div', 'lab-block', gi);
  C.wave = [1, 2].map((w) => {
    const r = el('div', 'lab-wave', C.wBlock);
    const row = el('div', 'hcc-row', r);
    el('span', 'hcc-label lab-wlabel', row, `k${w} =`);
    const key = 'k' + w;
    const comps = [0, 1, 2].map((c) => numIn(row, { label: ['', '', ''][c], value: S[key][c], step: 1, min: -32, max: 32, width: '3.6rem',
      title: ['k_x', 'k_y', 'k_z'][c] + ' (integer mode number)',
      onChange: (v) => { const k = S[key].slice(); k[c] = Math.round(v); app.setParam(key, k, 'ic'); } }));
    const amp = slider(r, { label: `A${w}`, min: 0, max: 3, step: 0.05, value: S['a' + w], onInput: set('a' + w, 'ic'), format: (v) => v.toFixed(2) });
    const ph = slider(r, { label: `φ${w} [°]`, min: 0, max: 360, step: 5, value: ((S['f' + w] % 360) + 360) % 360, onInput: set('f' + w, 'ic'), format: (v) => v.toFixed(0) });
    return { comps, amp, ph };
  });

  // peak block
  C.pBlock = el('div', 'lab-block', gi);
  C.pa = slider(C.pBlock, { label: 'amplitude', min: -3, max: 3, step: 0.05, value: S.pa, onInput: set('pa', 'ic'), format: (v) => v.toFixed(2) });
  C.pw = slider(C.pBlock, { label: 'width', min: 0.01, max: 0.3, step: 0.005, value: S.pw, onInput: set('pw', 'ic'), format: (v) => v.toFixed(3) });

  C.R = slider(gi, { label: 'smoothing R', min: 0.005, max: 0.1, value: S.R, log: true, onInput: set('R', 'ic'), format: (v) => v.toFixed(3) });
  C.sg = slider(gi, { label: 'σ0 (rms δ0)', min: 0.1, max: 3, step: 0.05, value: S.sg, onInput: set('sg', 'ic'), format: (v) => v.toFixed(2) });
  C.sp = slider(gi, { label: 'σ0 (0 = keep amp.)', min: 0, max: 3, step: 0.05, value: S.sp, onInput: set('sp', 'ic'), format: (v) => v.toFixed(2) });

  // ---------------------------------------------------------------- dynamics
  const gd = group(host, 'Dynamics');
  C.D = slider(gd, { label: 'growth D', min: 0, max: 2, step: 0.005, value: S.D, onInput: set('D', 'dyn'), format: (v) => v.toFixed(3) });
  const dscRow = el('div', 'lab-dsc', gd);
  C.dsc = el('span', null, dscRow, 'D_sc: –');
  C.om = sel(gd, { label: 'Ω_m', options: OMEGA_M.map((o) => [o, o === 1 ? '1 (EdS)' : String(o)]), value: S.om,
    title: 'Flat ΛCDM matter density. 1 = Einstein–de Sitter (exact Dⁿ growth); otherwise the LPT growth functions are integrated numerically. Changing it rebuilds the LPT (the IC is kept).',
    onChange: (v) => app.setParam('om', Number(v), 'all') });
  C.cosmo = el('div', 'lab-dsc', gd, 'a(D) = –');
  C.order = sel(gd, { label: 'LPT order', options: [[1, '1 (Zel’dovich)'], [2, '2 (2LPT)'], [3, '3 (3LPT)'], [4, '4 (4LPT)']], value: S.order, onChange: (v) => { app.setParam('order', Number(v), 'all'); C.syncVisibility(); } });
  C.growth = el('p', 'lab-note-line', gd, 'LPT growth g_τ(D)/Dⁿ: –');
  C.hs = sel(gd, { label: 'HC source ϕ', options: [['zel', 'Zel’dovich / Burgers (1LPT)'], ['lpt', 'nLPT longitudinal (Legendre transform)'], ['lptT', 'nLPT + transverse correction']], value: S.hs,
    title: 'Hopf–Cole source potential. Zel’dovich: S = −Dϕ, the Burgers equation in D. nLPT longitudinal: S = the longitudinal displacement potential of the chosen LPT order; the Legendre transform of q²/2 + S(q) inverts the gradient Lagrangian map, giving the exact nLPT Eulerian density before shell crossing. + transverse: first-order correction for the small curl part Ψ_T of 3LPT/4LPT (WASM only in 3D).',
    onChange: (v) => { app.setParam('hs', v, 'nu'); C.syncVisibility(); } });
  C.hsNote = el('p', 'lab-note-line', gd);
  C.hsNote.hidden = true;
  C.nu = slider(gd, { label: 'ν [L²]', min: 1e-6, max: 1e-2, value: S.nu, log: true, onInput: set('nu', 'nu'), format: (v) => v.toExponential(1) });
  const presets = el('div', 'hcc-row lab-nupresets', gd);
  el('span', 'hcc-label', presets, 'ν presets');
  for (const v of [1e-5, 1e-4, 1e-3, 1e-2]) {
    button(presets, { label: v.toExponential(0), onClick: () => { C.nu.set(v); app.setParam('nu', v, 'nu'); } });
  }
  C.nuInfo = el('p', 'lab-note-line', gd, 'shock width ν/(Δu·Δx): –');
  el('p', 'lab-note-line', gd, 'Shocks of width ν/Δu below one cell make the density (a derivative of u) noisy; below the grid floor the log-domain kernel degenerates into the discrete Hopf–Lax minimum.');
  C.me = sel(gd, { label: 'HC method', options: methodOptions(S.mode), value: S.me, onChange: (v) => { app.setParam('me', Number(v), 'nu'); C.syncVisibility(); } });
  C.mx = slider(gd, { label: 'max_exp', min: 10, max: 60, step: 1, value: S.mx, onInput: set('mx', 'nu'), format: (v) => v.toFixed(0) });
  C.hcNote = el('p', 'lab-note-line', gd);
  C.hcNote.hidden = true;
  const runRow = el('div', 'hcc-row lab-runrow', gd);
  C.live = checkbox(runRow, { label: 'live update', value: S.live, onChange: (v) => app.setParam('live', v, 'live') });
  C.run = button(runRow, { label: 'Run', onClick: () => app.runNow() });
  C.run.el.classList.add('lab-run');

  // ---------------------------------------------------------------- helpers
  C.syncVisibility = () => {
    const ic = S.ic;
    C.gBlock.hidden = ic !== 'g';
    C.wBlock.hidden = ic !== 'w';
    C.pBlock.hidden = ic !== 'p';
    C.pn.el.hidden = S.shape !== 'pl';
    C.ns.el.hidden = S.shape === 'pl';
    C.gm.el.hidden = S.shape === 'pl';
    C.sg.el.hidden = ic !== 'g';
    C.sp.el.hidden = ic === 'g';
    C.mx.el.hidden = S.me !== 0;
    C.wave.forEach((w) => { w.comps[2].show(S.mode === 3); });
    C.hsNote.hidden = S.hs === 'zel';
    C.hsNote.textContent = S.hs === 'zel' ? '' : `Legendre inversion of the order-${S.order} map` + (S.hs === 'lptT' ? (S.order > 2 ? ' with the transverse correction (rms Ψ_T/Ψ_L in the Legendre lab)' : ' (Ψ is a pure gradient through 2LPT: no correction)') : '')
      + (S.hs !== 'zel' && S.order === 1 ? ' = Zel’dovich' : '') + (S.mode === 3 && S.hs === 'lptT' && S.order > 2 ? '; WASM only in 3D, seconds per update' : S.mode === 3 && app.gpuCompute && S.gc ? '; the nLPT potential is computed by WASM and the solve runs on the GPU' : '') + '.';
    C.memWarn.hidden = !(S.mode === 3 && S.n >= 128);
    C.gc.el.hidden = !(S.mode === 3 && app.gpuCompute);
    C.gcNote.hidden = C.gc.el.hidden || !S.gc;
  };

  /** Reflect state S into all widgets (after a mode switch or hash restore). */
  C.syncAll = () => {
    C.modeRadios.forEach((r) => { r.checked = Number(r.value) === S.mode; });
    C.n.setOptions(gridOptions(S.mode), S.n);
    C.me.setOptions(methodOptions(S.mode), S.me);
    C.ic.set(S.ic); C.shape.set(S.shape);
    C.pn.set(S.pn); C.ns.set(S.ns); C.gm.set(S.gm); C.seed.set(S.seed);
    C.R.set(S.R); C.sg.set(S.sg); C.sp.set(S.sp);
    C.pa.set(S.pa); C.pw.set(S.pw);
    C.wave.forEach((w, i) => {
      const key = 'k' + (i + 1);
      w.comps.forEach((c, j) => c.set(S[key][j]));
      w.amp.set(S['a' + (i + 1)]); w.ph.set(((S['f' + (i + 1)] % 360) + 360) % 360);
    });
    C.D.set(S.D); C.om.set(S.om); C.order.set(S.order); C.hs.set(S.hs); C.nu.set(S.nu); C.me.set(S.me); C.mx.set(S.mx);
    C.live.set(S.live);
    C.gc.set(S.gc && app.gpuCompute);
    C.syncVisibility();
  };

  C.setDsc = (dsc, dsc1, D) => {
    C.dsc.innerHTML = `D<sub>sc</sub>(order ${S.order}) = <b class="${D > dsc ? 'lab-bad' : ''}">${fmtNum(dsc, 3)}</b> &nbsp; D<sub>sc</sub>(1) = <b class="${D > dsc1 ? 'lab-bad' : ''}">${fmtNum(dsc1, 3)}</b>`;
  };
  /** Show the effective viscosity actually used by the last Hopf–Cole solve. */
  C.setNuEff = (hc, nu) => {
    if (!hc) { C.hcNote.hidden = true; return; }
    const raised = hc.nuEff > nu * 1.001;
    C.hcNote.hidden = false;
    C.hcNote.className = 'lab-note-line' + (raised ? ' lab-bad' : '');
    C.hcNote.textContent = raised
      ? `ν_eff = ${fmtNum(hc.nuEff, 3)} (raised from ν: exponent range ${fmtNum(hc.range, 3)} exceeds max_exp)`
      : `ν_eff = ${fmtNum(hc.nuEff, 3)}, exponent range ${fmtNum(hc.range, 3)}`;
  };
  /** Cosmology readouts: a(D), Omega_m(a), D_max and the LPT growth ratios. */
  C.setCosmo = (cos, dmax, growth) => {
    if (!cos) { C.cosmo.textContent = 'a(D) = –'; C.growth.textContent = 'LPT growth g_τ(D)/Dⁿ: –'; return; }
    C.cosmo.innerHTML = `a(D) = <b>${fmtNum(cos.a, 4)}</b> &nbsp; Ω<sub>m</sub>(a) = <b>${fmtNum(cos.om, 3)}</b>` + (Number.isFinite(dmax) ? ` &nbsp; D<sub>max</sub> = ${fmtNum(dmax, 3)}` : ' &nbsp; (EdS: a = D; the constants −3/7, … are absorbed in the EdS term fields, so g/Dⁿ = 1)');
    C.growth.title = growth.map((q) => `${q.label}: ${q.full}`).join('\n');
    C.growth.textContent = 'LPT growth g_τ(D)/Dⁿ: ' + (growth.length ? growth.map((q) => `τ=${q.label}: ${fmtNum(q.g, 4)}`).join(' · ') : '–');
  };
  /** Restrict the D slider to 0.999 D_max in ΛCDM (2 in EdS). */
  C.setDMax = (dmax) => {
    const eff = Number.isFinite(dmax) ? Math.min(2, 0.999 * dmax) : 2;
    const input = C.D.el.querySelector('input');
    if (Number(input.max) !== eff) { input.max = eff; if (S.D > eff) S.D = eff; C.D.set(S.D); }
    return eff;
  };
  /** Shock width estimate (cells) and grid floor under the nu slider. */
  C.setNuInfo = (width, floor, nu, method) => {
    const w = Number.isFinite(width) ? fmtNum(width, 2) + ' cells' : '–';
    const fl = method === 0 ? 'none (Fourier multiplier)' : (Number.isFinite(floor) ? fmtNum(floor, 2) : '–');
    const below = Number.isFinite(floor) && method !== 0 && nu < floor;
    C.nuInfo.className = 'lab-note-line' + ((Number.isFinite(width) && width < 1) || below ? ' lab-bad' : '');
    C.nuInfo.textContent = `shock width ν/(Δu·Δx) = ${w} · grid floor ν_min = ${fl}${below ? ' (ν below floor)' : ''}`;
  };
  C.setDirty = (b) => {
    C.run.el.classList.toggle('lab-pending', b);
    C.run.setLabel(b ? 'Run (changes pending)' : 'Run');
  };

  C.syncVisibility();
  return C;
}
