// selftest.js - validation and benchmark of the WebGPU compute path against the WASM core.
// Used by gpu-test.html;  window.__gpuTest holds the machine-readable result after runAll().

import { loadCore } from '../hcc.js';
import { getGPU } from '../viz/gpu.js';
import { GpuFFT3D } from './fft3d.js';
import { GpuCosmo3D, dualDensityCpu } from './cosmo3d.js';
import { STORAGE_RW } from './util.js';
import { getTerms, getUnmergedSpec, growthUnmerged } from './terms.js';
import { GpuLpt3D } from './lpt3d.js';
import { GpuSheet3D } from './sheet3d.js';

const out = document.getElementById('out');
const tbl = document.getElementById('tbl');
const rows = [];
const log = (s) => { out.textContent += s + '\n'; };

function record(n, name, value, tol, fmt = (v) => v.toExponential(2)) {
  const pass = Number.isFinite(value) && value <= tol;
  rows.push({ n, name, value, tol, pass });
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${n}</td><td>${name}</td><td>${fmt(value)}</td><td>&lt; ${tol.toExponential(0)}</td><td class="${pass ? 'ok' : 'bad'}">${pass ? 'PASS' : 'FAIL'}</td>`;
  tbl.appendChild(tr);
}

const time = async (g, fn, reps = 5) => {
  const ts = [];
  for (let i = 0; i < reps + 1; i++) {
    const t0 = performance.now();
    const r = fn();
    if (r && r.then) await r;
    if (g) await g.sync();
    ts.push(performance.now() - t0);
  }
  ts.shift();                       // warm-up
  ts.sort((a, b) => a - b);
  return ts[Math.floor(ts.length / 2)];
};

export { getTerms };

const rms = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); };
const rmsDiff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s / a.length); };
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

function specRelErr(g, w) {
  if (g.k.length !== w.k.length) return Infinity;
  let m = 0;
  for (let i = 0; i < w.k.length; i++) {
    m = Math.max(m, Math.abs(g.p[i] - w.p[i]) / Math.abs(w.p[i]), Math.abs(g.k[i] - w.k[i]) / w.k[i], Math.abs(g.n[i] - w.n[i]));
  }
  return m;
}
function unpack(flat) { const nb = flat.length / 3; return { k: flat.slice(0, nb), p: flat.slice(nb, 2 * nb), n: flat.slice(2 * nb) }; }

/** q - x (box units, interleaved) from the WASM inverse map q (unwrapped), the input of GpuCosmo3D.dualDensity. */
function qToDq(q, n) {
  const N = n ** 3, o = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    const x = [Math.floor(i / (n * n)) / n, (Math.floor(i / n) % n) / n, (i % n) / n];
    for (let a = 0; a < 3; a++) o[3 * i + a] = q[3 * i + a] - x[a];
  }
  return o;
}
const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
const rmsM1 = (a) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - 1; s += d * d; } return Math.sqrt(s / a.length); };

/** Dual-sheet checks against hc_dual_density() of the last WASM hopf_cole / hopf_cole_lpt call: `dqGpu` = GPU-built dq buffer. */
async function checkDual(g, device, sim, n, tag, label, dualGpuBuf, tol) {
  const N = n ** 3;
  // reference: the f64 CPU implementation applied to the WASM inverse map (sim.hc_dual_density() sums the six Kuhn tetrahedra with
  // their alternating orientation and is not usable in 3-D); `wasm` in the labels below refers to this reference
  const dW = dualDensityCpu(sim.hc_qmap(), n);
  const rhoG = await g.readField(dualGpuBuf);
  record(tag, `dual sheet ${label}: GPU chain vs WASM, rms(rho_gpu - rho_wasm) / rms(rho_wasm - 1)`, rmsDiff(rhoG, dW) / rmsM1(dW), tol);
  record(tag, `dual sheet ${label}: |mean(rho_gpu) - 1|  (WASM ${mean(dW).toFixed(7)})`, Math.abs(mean(rhoG) - 1), 1e-5);
  // algorithm check: the WASM inverse map as input to the GPU shader
  const dqW = qToDq(sim.hc_qmap(), n);
  const bq = device.createBuffer({ size: 12 * N, usage: STORAGE_RW });
  device.queue.writeBuffer(bq, 0, dqW);
  const rhoA = await g.readField(g.dualDensity(bq));
  record(tag, `dual sheet ${label}, WASM q as input: rms(rho_gpu - rho_wasm) / rms(rho_wasm - 1)`, rmsDiff(rhoA, dW) / rmsM1(dW), 1e-4);
  bq.destroy();
}

async function makeSim(core, n, om = 1, order = 2) {
  const t0 = performance.now();
  const sim = new core.CosmoSim(3, n, 1);
  sim.set_ic_gaussian(0, -1, 0, 0.05, 1, 1);
  if (om !== 1) sim.set_cosmology(om);
  const t1 = performance.now();
  sim.build_lpt(order);
  return { sim, tIC: t1 - t0, tLPT: performance.now() - t1 };
}

export async function validate(core, gpu, n, { om = 1, order = 2 } = {}) {
  const { device } = gpu;
  const tag = om === 1 ? `${n}` : `${n} (Om=${om}, ${order}LPT)`;
  log(`--- validation n = ${tag}`);
  const { sim } = await makeSim(core, n, om, order);
  const N = n ** 3;
  const T = getTerms(sim, n);
  log(`terms: ${T.orders.length} (orders ${T.orders.join(',')}) via ${T.source}`);
  const g = new GpuCosmo3D(device, n); await g.init();
  await g.uploadTerms(T.terms, T.orders);
  await g.setPhi0(sim.phi0());
  const D = 0.3, nu = 1e-4;

  // positions
  const posB = g.positions(T.gvals(D), order);
  const posG = await g.readField(posB, 3 * N);
  const posW = sim.positions(D, order);
  record(tag, 'positions: max |x_gpu - x_wasm|', maxDiff(posG, posW), 2e-6);

  // CIC
  const rhoB = g.cicDensity(posB);
  const rhoG = await g.readField(rhoB);
  const rhoW = sim.cic_density(D, order, n);
  let mean = 0; for (let i = 0; i < N; i++) mean += rhoG[i]; mean /= N;
  record(tag, 'CIC: rms(1+d_gpu - 1+d_wasm) / rms(1+d)', rmsDiff(rhoG, rhoW) / rms(rhoW), 1e-4);
  record(tag, 'CIC: |mean(rho) - 1|', Math.abs(mean - 1), 1e-6);

  // Hopf-Cole
  sim.hopf_cole(D, nu, 1, 30);
  const dW = sim.hc_delta();
  const hc = g.hopfCole(nu, D);
  const dG = await g.readField(hc.delta);
  record(tag, `Hopf-Cole d (nu=1e-4, D=0.3, w=${hc.w}): rms diff / rms(d)`, rmsDiff(dG, dW) / rms(dW), 1e-3);
  record(tag, 'Hopf-Cole d: max |diff| / rms(d)', maxDiff(dG, dW) / rms(dW), 1e-2);
  // a second, more nonlinear point (D = 1, nu = 1e-3)
  sim.hopf_cole(1.0, 1e-3, 1, 30);
  const dW2 = sim.hc_delta();
  const hc2 = g.hopfCole(1e-3, 1.0);
  const dG2 = await g.readField(hc2.delta);
  record(tag, `Hopf-Cole d (nu=1e-3, D=1, w=${hc2.w}): rms diff / rms(d)`, rmsDiff(dG2, dW2) / rms(dW2), 1e-3);

  // dual sheet (Zel'dovich source): q = x - D grad Phi_v on the GPU
  for (const [Dd, nud] of [[D, nu], [1.0, 1e-3]]) {
    const enc = device.createCommandEncoder();
    g.hopfCole(nud, Dd, enc);
    const dq = g.hcDisplacement(Dd, enc);
    g.dualDensity(dq, enc);
    device.queue.submit([enc.finish()]);
    sim.hopf_cole(Dd, nud, 1, 30);
    await checkDual(g, device, sim, n, tag, `Zel'dovich (D=${Dd}, nu=${nud})`, g.buf('dual-rho'), 1e-4);
  }

  // spectra of the CIC delta (deconvolved and plain) and of the Hopf-Cole delta
  const deltaCic = Float32Array.from(rhoW, (v) => v - 1);
  const fieldB = g.uploadField(deltaCic);
  const nb = 40;
  const an = await g.analyze(fieldB, { nbins: nb, maps: true });
  record(tag, 'P(k) CIC deconvolved: max bin rel. error', specRelErr(an.dec, unpack(sim.power_spectrum(deltaCic, nb, true))), 1e-3);
  record(tag, 'P(k) CIC plain: max bin rel. error', specRelErr(an.plain, unpack(sim.power_spectrum(deltaCic, nb, false))), 1e-3);
  const ps = await g.powerSpectrum(g.uploadField(dW), nb, false);
  record(tag, 'P(k) Hopf-Cole (nu=1e-4, D=1 field): max bin rel. error', specRelErr(ps, unpack(sim.power_spectrum(dW, nb, false))), 1e-3);
  // GPU delta through the GPU spectrum vs WASM (end-to-end, GPU-generated field)
  const hc3 = g.hopfCole(nu, D);
  const psG = await g.powerSpectrum(hc3.delta, nb, false);
  sim.hopf_cole(D, nu, 1, 30);
  const wref = unpack(sim.power_spectrum(sim.hc_delta(), nb, false));
  record(tag, 'P(k) of GPU HC field vs WASM HC field: max bin rel. error', specRelErr(psG, wref), 2e-3);

  // maps (compare where the reference amplitude is visible: > -6)
  const ampW = sim.fourier_amp(deltaCic), phW = sim.fourier_phase(deltaCic);
  let ma = 0; for (let i = 0; i < ampW.length; i++) if (ampW[i] > -6) ma = Math.max(ma, Math.abs(an.maps.amp[i] - ampW[i]));
  record(tag, 'Fourier amp (where > -6): max |diff|', ma, 5e-3);
  let mp = 0; for (let i = 0; i < ampW.length; i++) if (ampW[i] > -3) { let d = Math.abs(an.maps.phase[i] - phW[i]); d = Math.min(d, 2 * Math.PI - d); mp = Math.max(mp, d); }
  record(tag, 'Fourier phase (where amp > -3): max |diff| [rad]', mp, 5e-3);

  // cross spectrum with a reference field
  const d0 = sim.linear_delta(1.0);
  await g.setReference(d0, nb);
  const anx = await g.analyze(g.uploadField(deltaCic), { nbins: nb, cross: true });
  const cw = unpack(sim.cross_spectrum(deltaCic, d0, nb));
  const p0w = unpack(sim.power_spectrum(d0, nb, false));
  const pfw = unpack(sim.power_spectrum(deltaCic, nb, false));
  let me = 0;
  const p0max = Math.max(...p0w.p);
  // bins where the reference has < 1e-8 of its peak power are at the f32 FFT noise floor (the CPU reference is f64)
  for (let i = 0; i < cw.k.length; i++) if (p0w.p[i] > 1e-8 * p0max) me = Math.max(me, Math.abs(anx.cross.p[i] - cw.p[i]) / Math.sqrt(p0w.p[i] * pfw.p[i]));
  record(tag, 'cross spectrum with delta0 (bins with P0 > 1e-8 max): max |dX| / sqrt(P_f P_0)', me, 2e-3);

  // FFT round trip and plane wave
  const fft = new GpuFFT3D(device, n); await fft.init();
  const rnd = new Float32Array(N); let s = 12345;
  for (let i = 0; i < N; i++) { s = (s * 1664525 + 1013904223) >>> 0; rnd[i] = s / 4294967296 - 0.5; }
  const mk = (bytes) => device.createBuffer({ size: bytes, usage: STORAGE_RW });
  const bR = mk(4 * N), bC1 = mk(8 * N), bC2 = mk(8 * N), bC3 = mk(8 * N), bR2 = mk(4 * N);
  device.queue.writeBuffer(bR, 0, rnd);
  const enc = device.createCommandEncoder();
  fft.packReal(bR, bC1, enc); fft.forward(bC1, bC2, enc); fft.inverse(bC2, bC3, enc); fft.unpackReal(bC3, bR2, enc);
  device.queue.submit([enc.finish()]);
  const back = await g.readField(bR2);
  record(tag, 'FFT round trip: max |ifft(fft(f)) - f|', maxDiff(back, rnd), 1e-5);
  // Parseval / direct DFT spot check against the WASM spectrum is covered by P(k); plane wave:
  const pw = new Float32Array(N);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) pw[(i * n + j) * n + k] = Math.cos(2 * Math.PI * 3 * i / n);
  device.queue.writeBuffer(bR, 0, pw);
  const enc2 = device.createCommandEncoder();
  fft.packReal(bR, bC1, enc2); fft.forward(bC1, bC2, enc2);
  device.queue.submit([enc2.finish()]);
  const spec = await g.readField(bC2, 2 * N);
  let off = 0, a3 = 0, am3 = 0;
  for (let q = 0; q < N; q++) {
    const i = Math.floor(q / (n * n)), rest = q % (n * n);
    const mag = Math.hypot(spec[2 * q], spec[2 * q + 1]);
    if (rest === 0 && i === 3) a3 = mag; else if (rest === 0 && i === n - 3) am3 = mag; else off = Math.max(off, mag);
  }
  record(tag, 'plane wave cos(2pi 3x): leakage max|other| / (N/2)', off / (N / 2), 1e-5);
  record(tag, 'plane wave: |amp(+3) - N/2|/(N/2) and |amp(-3) - N/2|/(N/2)', Math.max(Math.abs(a3 - N / 2), Math.abs(am3 - N / 2)) / (N / 2), 1e-5);
  [bR, bC1, bC2, bC3, bR2].forEach((b) => b.destroy());
  fft.destroy(); g.destroy(); sim.free();
}

export async function bench(core, gpu, n) {
  const { device } = gpu;
  log(`--- benchmark n = ${n}`);
  const { sim, tIC, tLPT } = await makeSim(core, n);
  log(`WASM IC ${tIC.toFixed(0)} ms, build_lpt(2) ${tLPT.toFixed(0)} ms`);
  const N = n ** 3;
  const T = getTerms(sim, n);
  const g = new GpuCosmo3D(device, n); await g.init();
  const tUp0 = performance.now();
  await g.uploadTerms(T.terms, T.orders);
  await g.setPhi0(sim.phi0());
  await g.sync();
  const tUp = performance.now() - tUp0;
  const D = 0.3, nu = 1e-4;
  const R = { n, upload: tUp, wasmIC: tIC, wasmLPT: tLPT };
  const gv = T.gvals(D);
  R.gpu_positions = await time(g, () => g.positions(gv, 2));
  const posB = g.positions(gv, 2);
  R.gpu_cic = await time(g, () => g.cicDensity(posB));
  R.gpu_cic_readback = await time(null, async () => { await g.readField(g.cicDensity(posB)); });
  R.gpu_hc = await time(g, () => g.hopfCole(nu, D));
  R.gpu_hc_readback = await time(null, async () => { await g.readField(g.hopfCole(nu, D).delta); });
  const hc = g.hopfCole(nu, D);
  const fieldB = hc.delta;
  const fft = g.fft;
  const cA = g.buf('t-a', 8 * N), cB = g.buf('t-b', 8 * N);
  R.gpu_fft = await time(g, () => { const e = device.createCommandEncoder(); fft.forward(cA, cB, e); device.queue.submit([e.finish()]); });
  R.gpu_pk_all = await time(null, async () => { await g.analyze(fieldB, { nbins: 40 }); });
  R.gpu_pk_maps = await time(null, async () => { await g.analyze(fieldB, { nbins: 40, maps: true }); });
  R.gpu_upload_field = await time(g, () => g.uploadField(new Float32Array(N)));
  R.gpu_hc_window = hc.w;
  // WASM
  const reps = n >= 128 ? 1 : 2;
  const tw = (fn) => { const ts = []; for (let i = 0; i < reps; i++) { const t0 = performance.now(); fn(); ts.push(performance.now() - t0); } return Math.min(...ts); };
  R.wasm_positions = tw(() => sim.positions(D, 2));
  let rho; R.wasm_cic = tw(() => { rho = sim.cic_density(D, 2, n); });
  R.wasm_hc = tw(() => sim.hopf_cole(D, nu, 1, 30));
  const dW = sim.hc_delta();
  R.wasm_pk = tw(() => sim.power_spectrum(dW, 40, false));
  R.wasm_maps = tw(() => sim.fourier_maps(dW));
  // GPU vs WASM end-to-end check on the benchmarked field
  const dG = await g.readField(hc.delta);
  R.hc_rms_diff_rel = rmsDiff(dG, dW) / rms(dW);
  log(JSON.stringify(R, (k, v) => (typeof v === 'number' ? +v.toPrecision(4) : v)));
  g.destroy(); sim.free();
  return R;
}


// ------------------------------------------------------------------------------------------------------------------
// GPU-resident nLPT: term build, D_sc, Legendre (+ transverse), tetrahedral sheet

const relRms = (a, b) => rmsDiff(a, b) / Math.max(rms(b), 1e-30);

/** Interleaved xyz field from the three WASM component arrays. */
function interleave(f, N) { const o = new Float32Array(3 * N); for (let c = 0; c < 3; c++) { const a = f(c); for (let i = 0; i < N; i++) o[3 * i + c] = a[i]; } return o; }

export async function validateLpt(core, gpu, n, { om = 1, order = 4 } = {}) {
  const { device } = gpu;
  const tag = om === 1 ? `${n} (EdS, ${order}LPT)` : `${n} (Om=${om}, ${order}LPT)`;
  log(`--- GPU nLPT validation n = ${tag}`);
  const N = n ** 3;
  const { sim } = await makeSim(core, n, om, order);
  const spec = getUnmergedSpec(sim);
  const d0 = sim.linear_delta(1.0);
  const g = new GpuCosmo3D(device, n); await g.init();
  const lpt = new GpuLpt3D(g); await lpt.init();
  const sheet = new GpuSheet3D(g); await sheet.init();
  const tb = await lpt.build(d0, spec, order);
  log(`built ${tb.count} raw terms in ${tb.ms.toFixed(0)} ms`);
  const gf = (D) => growthUnmerged(sim, spec, D).slice(0, tb.count);
  if (om !== 1) {
    // the lab uses the cheap table lookup term_g in LCDM; it must equal the unmerged accessor
    let md = 0; for (const D of [0.1, 0.3, 0.6]) { const a = sim.term_g(D), b = sim.term_g_unmerged(D); for (let t = 0; t < a.length; t++) md = Math.max(md, Math.abs(a[t] - b[t]) / Math.max(1e-30, Math.abs(b[t]))); }
    record(tag, 'term_g(D) (tables) vs term_g_unmerged(D): max relative difference', md, 1e-9);
  }

  // --- term fields
  const dev = [];
  if (om === 1) {
    const S = [];
    for (let t = 0; t < tb.count; t++) S.push(await g.readField(g.terms[t], 3 * N));
    for (let o = 1; o <= order; o++) {
      const sum = new Float32Array(3 * N);
      for (let t = 0; t < tb.count; t++) if (spec.orders[t] === o) { const c = spec.coefs[t]; for (let i = 0; i < 3 * N; i++) sum[i] += c * S[t][i]; }
      const ref = interleave((c) => sim.term_psi(o - 1, c), N);
      record(tag, `order ${o}: rel rms of sum_tau c_tau S^tau vs term_psi`, relRms(sum, ref), 1e-4);
    }
  } else {
    for (let t = 0; t < tb.count; t++) {
      const S = await g.readField(g.terms[t], 3 * N);
      const ref = interleave((c) => sim.term_psi(t, c), N);
      dev.push(relRms(S, ref));
    }
    record(tag, `${tb.count} raw terms: worst rel rms vs term_psi`, Math.max(...dev), 1e-4);
  }
  record(tag, 'phi0 (GPU, from delta0): rel rms vs WASM phi0', relRms(await g.readField(g.buf('phi0')), sim.phi0()), 1e-4);

  // --- D_sc (EdS: D_sc is where the first Jacobian vanishes; compare the whole search)
  const dmax = sim.d_max();
  const dsc = [];
  for (let o = 1; o <= order; o++) {
    const gpuD = await lpt.shellCrossing(o, (D) => gf(D), dmax);
    const wD = sim.shell_crossing(o);
    dsc.push(wD);
    record(tag, `D_sc(${o}LPT) GPU ${gpuD.toFixed(5)} vs WASM ${wD.toFixed(5)}: relative difference`, Number.isFinite(wD) ? Math.abs(gpuD - wD) / wD : (gpuD === wD ? 0 : Infinity), 1e-3);
  }

  // --- positions
  const Dp = 0.3;
  for (let o = 1; o <= order; o++) {
    const px = await g.readField(g.positions(gf(Dp), o), 3 * N);
    const pw = sim.positions(Dp, o);
    const q = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { const ix = Math.floor(i / (n * n)), iy = Math.floor(i / n) % n, iz = i % n; q[3 * i] = ix / n; q[3 * i + 1] = iy / n; q[3 * i + 2] = iz / n; }
    const dg = new Float32Array(3 * N), dw = new Float32Array(3 * N);
    for (let i = 0; i < 3 * N; i++) { dg[i] = px[i] - q[i]; dw[i] = pw[i] - q[i]; }
    record(tag, `positions ${o}LPT (D=${Dp}): rms|x_gpu - x_wasm| / rms(Psi)`, relRms(dg, dw), 1e-4);
  }

  // --- Legendre inversion (two operating points: before and after the first shell crossing of the chosen order)
  const dscO = Number.isFinite(dsc[order - 1]) ? dsc[order - 1] : 1;
  const cases = [{ name: '0.5 D_sc', D: 0.5 * dscO, nu: 1e-4 }, { name: '1.5 D_sc', D: 1.5 * dscO, nu: 1e-3 }, { name: '3 D_sc', D: 3 * dscO, nu: 1e-3 }];
  for (const cs of cases) {
    const { D, nu } = cs;
    const gv = gf(D);
    const L = await lpt.legendre(gv, order, D);
    let dNoT = null;
    for (const tr of (order > 2 ? [false, true] : [false])) {
      const enc = device.createCommandEncoder();
      const hc = g.hopfCole(nu, D, enc, 'phi-eff');
      if (tr) lpt.transverse(D, enc);
      device.queue.submit([enc.finish()]);
      const dG = await g.readField(hc.delta);
      const frac = sim.hopf_cole_lpt(D, order, nu, 1, 30, tr);
      const dW = sim.hc_delta();
      record(tag, `Legendre ${order}LPT${tr ? '+transverse' : ''} (${cs.name}=${D.toFixed(4)}, nu=${nu}, w=${hc.w}): rms diff / rms(d)`, relRms(dG, dW), 1e-3);
      {
        const enc2 = device.createCommandEncoder();
        g.hopfCole(nu, D, enc2, 'phi-eff');
        const dq = tr ? lpt.transverse(D, enc2) : g.hcDisplacement(D, enc2);
        g.dualDensity(dq, enc2);
        device.queue.submit([enc2.finish()]);
        sim.hopf_cole_lpt(D, order, nu, 1, 30, tr);
        await checkDual(g, device, sim, n, tag, `Legendre ${order}LPT${tr ? '+transverse' : ''} (${cs.name})`, g.buf('dual-rho'), cs.D < dscO ? 1e-4 : 1e-3);
      }
      if (!tr) dNoT = dG;
      else if (dNoT) log(`  size of the transverse correction at ${cs.name}: rms(d_T - d_noT)/rms(d) = ${relRms(dG, dNoT).toExponential(2)}`);
      if (tr) record(tag, `  rms|Psi_T|/rms|Psi_L| GPU ${L.frac.toExponential(3)} vs WASM ${frac.toExponential(3)}: rel. difference`, Math.abs(L.frac - frac) / Math.max(frac, 1e-30), 1e-3);
    }
    // --- tetrahedral sheet
    const disp = g.displacement(gv, order);
    const rhoG = await g.readField(sheet.density(disp));
    const rhoW = sim.sheet_density(D, order, n, 1);
    let mg = 0, mw = 0; for (let i = 0; i < N; i++) { mg += rhoG[i]; mw += rhoW[i]; }
    const rhoRef = Float32Array.from(rhoW, (v) => v - 1);
    // before the first shell crossing the density is smooth and the f32 terms are enough for 1e-4; after it the density is dominated
    // by caustics, which amplify the 1e-6 relative differences of the f32 vs f64 displacement: end-to-end tolerance 1e-3 there
    record(tag, `sheet ${order}LPT (${cs.name}): rms(rho_gpu - rho_wasm) / rms(rho_wasm - 1)`, rmsDiff(rhoG, rhoW) / rms(rhoRef), cs.D < dscO ? 1e-4 : 1e-3);
    if (cs.D >= dscO) {
      // algorithm check: the same tetrahedra (WASM positions as input) must reproduce the f64 raster to 1e-4 also with multiple streams
      const pw = sim.positions(D, order), psiW = new Float32Array(3 * N);
      for (let i = 0; i < N; i++) { const ix = Math.floor(i / (n * n)), iy = Math.floor(i / n) % n, iz = i % n; psiW[3 * i] = pw[3 * i] - ix / n; psiW[3 * i + 1] = pw[3 * i + 1] - iy / n; psiW[3 * i + 2] = pw[3 * i + 2] - iz / n; }
      const bW = device.createBuffer({ size: 12 * N, usage: STORAGE_RW });
      device.queue.writeBuffer(bW, 0, psiW);
      const rhoA = await g.readField(sheet.density(bW));
      record(tag, `sheet ${order}LPT (${cs.name}), WASM positions as input: rms(rho_gpu - rho_wasm) / rms(rho_wasm - 1)`, rmsDiff(rhoA, rhoW) / rms(rhoRef), 1e-4);
      bW.destroy();
    }
    record(tag, `sheet ${order}LPT (${cs.name}): |mean_gpu - mean_wasm|  (means ${(mg / N).toFixed(6)} / ${(mw / N).toFixed(6)})`, Math.abs(mg - mw) / N, 1e-5);
  }
  // velocity jump
  const du = await lpt.velocityJump();
  const v = sim.velocities(1.0, 1);
  let lo = Infinity, hi = 0; for (let i = 0; i < N; i++) { const s = Math.hypot(v[3 * i], v[3 * i + 1], v[3 * i + 2]); lo = Math.min(lo, s); hi = Math.max(hi, s); }
  record(tag, 'velocity jump max|S1| - min|S1|: relative difference', Math.abs(du - (hi - lo)) / (hi - lo), 1e-4);
  lpt.destroy(); g.destroy(); sim.free();
}

const msOf = async (g, fn, reps = 3) => {
  const ts = [];
  for (let i = 0; i < reps + 1; i++) { const t0 = performance.now(); const r = fn(); if (r && r.then) await r; if (g) await g.sync(); ts.push(performance.now() - t0); }
  ts.shift(); ts.sort((a, b) => a - b); return ts[Math.floor(ts.length / 2)];
};

/** Timings of every stage of the 3-D nLPT pipeline, GPU vs WASM. `wasmOrder` limits the WASM reference at large n. */
export async function benchLpt(core, gpu, n, { order = 4, wasmOrder = order, wasm = true } = {}) {
  const { device } = gpu;
  log(`--- nLPT pipeline benchmark n = ${n}, GPU order ${order}, WASM order ${wasmOrder}`);
  const N = n ** 3;
  const R = { n, order, wasmOrder };
  let t0 = performance.now();
  const sim = new core.CosmoSim(3, n, 1);
  sim.set_ic_gaussian(0, -1, 0, 0.05, 1, 1);
  const d0 = sim.linear_delta(1.0);
  R.IC = performance.now() - t0;
  // helper sim for the growth tables / term list (what the lab does)
  const helper = new core.CosmoSim(3, 8, 1); helper.build_lpt(order);
  const spec = getUnmergedSpec(helper);
  const g = new GpuCosmo3D(device, n); await g.init();
  const lpt = new GpuLpt3D(g); await lpt.init();
  const sheet = new GpuSheet3D(g); await sheet.init();
  t0 = performance.now();
  await lpt.build(d0, spec, order);
  await g.sync();
  R.gpu_build_first = performance.now() - t0;
  R.gpu_build = await msOf(g, () => lpt.build(d0, spec, order), 2);
  const gf = (D) => growthUnmerged(helper, spec, D).slice(0, spec.count);
  const dmax = Infinity;
  t0 = performance.now();
  const dscG = await lpt.shellCrossing(order, gf, dmax);
  R.gpu_dsc = performance.now() - t0;
  R.gpu_dsc = await msOf(null, () => lpt.shellCrossing(order, gf, dmax), 2);
  R.dsc = dscG;
  const D = 0.5 * dscG, nu = 1e-4;
  const gv = gf(D);
  R.gpu_positions = await msOf(g, () => g.positions(gv, order));
  R.gpu_cic = await msOf(g, () => g.cicDensity(g.positions(gv, order)));
  R.gpu_hc_zel = await msOf(g, () => g.hopfCole(nu, D));
  R.gpu_legendre_split = await msOf(null, () => lpt.legendre(gv, order, D));
  R.gpu_legendre_hc = await msOf(g, () => g.hopfCole(nu, D, null, 'phi-eff'));
  R.gpu_legendre_total = await msOf(null, async () => { await lpt.legendre(gv, order, D); const e = device.createCommandEncoder(); g.hopfCole(nu, D, e, 'phi-eff'); lpt.transverse(D, e); device.queue.submit([e.finish()]); });
  R.gpu_legendre_total_readback = await msOf(null, async () => { await lpt.legendre(gv, order, D); const e = device.createCommandEncoder(); const hc = g.hopfCole(nu, D, e, 'phi-eff'); lpt.transverse(D, e); device.queue.submit([e.finish()]); await g.readField(hc.delta); });
  R.gpu_sheet = await msOf(g, () => sheet.density(g.displacement(gv, order)));
  R.gpu_sheet_readback = await msOf(null, async () => { await g.readField(sheet.density(g.displacement(gv, order))); });
  R.gpu_dpath = await msOf(null, async () => { const pos = g.positions(gf(D * 1.01), order); await g.readField(g.cicDensity(pos)); });
  if (wasm) {
    const wsim = sim;
    t0 = performance.now(); wsim.build_lpt(wasmOrder); R.wasm_build = performance.now() - t0;
    const tw = (fn) => { const t = performance.now(); fn(); return performance.now() - t; };
    R.wasm_positions = tw(() => wsim.positions(D, wasmOrder));
    R.wasm_cic = tw(() => wsim.cic_density(D, wasmOrder, n));
    R.wasm_hc_zel = tw(() => wsim.hopf_cole(D, nu, 1, 30));
    R.wasm_legendre_transverse = tw(() => wsim.hopf_cole_lpt(D, wasmOrder, nu, 1, 30, true));
    R.wasm_dsc = tw(() => { R.wasm_dsc_value = wsim.shell_crossing(wasmOrder); });
    R.wasm_sheet = tw(() => wsim.sheet_density(D, wasmOrder, n, 1));
  }
  log(JSON.stringify(R, (k, v) => (typeof v === 'number' ? +v.toPrecision(4) : v)));
  lpt.destroy(); g.destroy(); sim.free(); helper.free();
  return R;
}

export async function runAll({ val = [32, 64], ben = [64, 128], lval = [32, 64], lben = [64, 128], wasmOrder128 = 4, skipOld = false } = {}) {
  const res = { rows, bench: [], lbench: [], error: null };
  window.__gpuTest = res;
  try {
    const core = await loadCore();
    const gpu = await getGPU();
    if (!gpu) { log('WebGPU not available'); res.error = 'no webgpu'; return res; }
    log(`adapter limits: maxStorageBufferBindingSize ${(gpu.device.limits.maxStorageBufferBindingSize / 1048576)} MiB, maxBufferSize ${(gpu.device.limits.maxBufferSize / 1048576)} MiB`);
    if (!skipOld) {
      for (const n of val) await validate(core, gpu, n);
      await validate(core, gpu, 32, { om: 0.3, order: 3 });          // flat LCDM: 8 LPT terms, numerically integrated growth
    }
    for (const n of lval) await validateLpt(core, gpu, n, { om: 1, order: 4 });
    for (const n of lval) await validateLpt(core, gpu, n, { om: 0.3, order: 3 });
    if (!skipOld) for (const n of ben) res.bench.push(await bench(core, gpu, n));
    for (const n of lben) res.lbench.push(await benchLpt(core, gpu, n, { order: 4, wasmOrder: n >= 128 ? wasmOrder128 : 4, wasm: !(n >= 128 && wasmOrder128 === 0) }));
    const fails = rows.filter((r) => !r.pass).length;
    log(fails ? `FAILED: ${fails} of ${rows.length}` : `ALL ${rows.length} CHECKS PASSED`);
    res.fails = fails;
  } catch (e) {
    console.error(e);
    log('ERROR: ' + (e && e.stack ? e.stack : e));
    res.error = String(e && e.message ? e.message : e);
  }
  res.done = true;
  return res;
}

const q = new URLSearchParams(location.search);
const list = (k, d) => (q.has(k) ? q.get(k).split(',').filter(Boolean).map(Number) : d);
const opts = () => ({ val: list('val', [32, 64]), ben: list('bench', [64, 128]), lval: list('lval', [32, 64]), lben: list('lbench', [64, 128]), wasmOrder128: list('wo128', [4])[0], skipOld: q.has('skipold') });
document.getElementById('run').addEventListener('click', () => runAll(opts()));
if (q.has('auto')) runAll(opts());
