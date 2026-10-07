// selftest.js - validation and benchmark of the WebGPU compute path against the WASM core.
// Used by gpu-test.html;  window.__gpuTest holds the machine-readable result after runAll().

import { loadCore } from '../hcc.js';
import { getGPU } from '../viz/gpu.js';
import { GpuFFT3D } from './fft3d.js';
import { GpuCosmo3D } from './cosmo3d.js';
import { STORAGE_RW } from './util.js';
import { getTerms } from './terms.js';

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
  g.setPhi0(sim.phi0());
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
  g.setPhi0(sim.phi0());
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

export async function runAll({ val = [32, 64], ben = [64, 128] } = {}) {
  const res = { rows, bench: [], error: null };
  window.__gpuTest = res;
  try {
    const core = await loadCore();
    const gpu = await getGPU();
    if (!gpu) { log('WebGPU not available'); res.error = 'no webgpu'; return res; }
    log(`adapter limits: maxStorageBufferBindingSize ${(gpu.device.limits.maxStorageBufferBindingSize / 1048576)} MiB, maxBufferSize ${(gpu.device.limits.maxBufferSize / 1048576)} MiB`);
    for (const n of val) await validate(core, gpu, n);
    await validate(core, gpu, 32, { om: 0.3, order: 3 });          // flat LCDM: 8 LPT terms, numerically integrated growth
    for (const n of ben) res.bench.push(await bench(core, gpu, n));
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
document.getElementById('run').addEventListener('click', () => runAll({ val: list('val', [32, 64]), ben: list('bench', [64, 128]) }));
if (q.has('auto')) runAll({ val: list('val', [32, 64]), ben: list('bench', [64, 128]) });
