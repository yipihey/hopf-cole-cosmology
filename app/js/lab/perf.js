// perf.js - capability probing, performance presets (auto / lite / full), 3D memory estimate and the diagnostics text.
//
// Auto rule (decidePerf): lite when ANY of
//   * WebGPU is unavailable (disabled, missing, adapter/device request failed or timed out, device lost)
//   * adapter.info.isFallbackAdapter
//   * vendor / architecture / device / description matches /intel|llvmpipe|swiftshader|basic render/i
//   * adapter maxStorageBufferBindingSize < 128 MiB
//   * navigator.hardwareConcurrency <= 4  or  navigator.deviceMemory <= 4
//   * the 64^3 GPU FFT micro-benchmark (forward + inverse) takes > 150 ms
// otherwise full.  `?lite=1` and perf=lite force lite, perf=full forces full.

import { GpuFFT3D } from '../gpu/fft3d.js';
import { withTimeout } from '../viz/gpu.js';

const MiB = 1048576, GiB = 1073741824;
export const GPU_NAME_RE = /intel|llvmpipe|swiftshader|basic render/i;
export const BENCH_LIMIT_MS = 150;

/** Lite restrictions. */
export const LITE_GRID_3D = [32, 48, 64];
export const LITE_GRID_2D = [128, 256];

export function queryFlags() {
  const q = new URLSearchParams(location.search);
  const on = (k) => q.has(k) && !['0', 'false', 'off'].includes(q.get(k));
  return { lite: on('lite'), nogpu: q.has('nogpu'), gpufail: q.get('gpufail'), gpuslow: on('gpuslow') };
}

/** Facts that are known without touching the GPU. */
export function hostFacts() {
  const nav = typeof navigator !== 'undefined' ? navigator : {};
  return { hardwareConcurrency: nav.hardwareConcurrency, deviceMemory: nav.deviceMemory, userAgent: nav.userAgent || '' };
}

/** Sync-only reasons for lite (used before the GPU probe has finished). */
export function syncLiteReasons(flags) {
  const r = [];
  const h = hostFacts();
  if (flags.nogpu) r.push('WebGPU disabled by ?nogpu');
  else if (typeof navigator === 'undefined' || !navigator.gpu) r.push('WebGPU not supported by this browser');
  if (h.hardwareConcurrency !== undefined && h.hardwareConcurrency <= 4) r.push(`hardwareConcurrency = ${h.hardwareConcurrency} (<= 4)`);
  if (h.deviceMemory !== undefined && h.deviceMemory <= 4) r.push(`deviceMemory = ${h.deviceMemory} GB (<= 4)`);
  return r;
}

/**
 * Decide the preset.
 * @param pref 'auto' | 'lite' | 'full' ; @param flags queryFlags() ; @param gpu gpuStatus() snapshot or null (not yet probed) ; @param bench {ms} | null
 * @returns {{lite:boolean, reason:string, reasons:string[], source:string}}
 */
export function decidePerf(pref, flags, gpu, bench) {
  if (flags.lite) return { lite: true, source: 'query', reasons: ['?lite=1'], reason: 'forced by ?lite=1' };
  if (pref === 'lite') return { lite: true, source: 'user', reasons: ['selected'], reason: 'selected by the user' };
  if (pref === 'full') return { lite: false, source: 'user', reasons: ['selected'], reason: 'selected by the user' };
  const r = syncLiteReasons(flags);
  if (gpu) {
    if (!gpu.available) { if (!r.some((x) => /WebGPU/.test(x))) r.push('WebGPU unavailable: ' + gpu.reason); }
    else {
      const i = gpu.info || {};
      if (i.isFallbackAdapter) r.push('software fallback adapter');
      const names = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' / ');
      if (GPU_NAME_RE.test(names)) r.push(`integrated/software GPU (${names})`);
      const mx = gpu.limits && gpu.limits.maxStorageBufferBindingSize;
      if (mx !== undefined && mx < 128 * MiB) r.push(`maxStorageBufferBindingSize ${(mx / MiB).toFixed(0)} MiB < 128 MiB`);
      if (bench && bench.ms > BENCH_LIMIT_MS) r.push(`GPU micro-benchmark ${bench.ms.toFixed(0)} ms > ${BENCH_LIMIT_MS} ms`);
    }
  }
  const lite = r.length > 0;
  return { lite, source: 'auto', reasons: r, reason: lite ? r.join('; ') : 'capable GPU, benchmark ' + (bench ? bench.ms.toFixed(0) + ' ms' : 'n/a') };
}

/**
 * ~300 ms micro-benchmark: one 64^3 GPU FFT forward + inverse through GpuFFT3D. The first (cold) run includes driver warm-up, the
 * second (warm) is the reported time unless the cold run alone is > 1 s. `?gpuslow=1` reports 500 ms (test hook).
 * Never throws; resolves {ms, cold, init, error?}.
 */
export async function runBench(device, flags = {}) {
  const out = { ms: NaN, cold: NaN, init: NaN, error: null, simulated: false };
  const t0 = performance.now();
  try {
    await withTimeout((async () => {
      const n = 64, size = n * n * n;
      const fft = new GpuFFT3D(device, n);
      const ti = performance.now();
      await fft.init();
      out.init = performance.now() - ti;
      const mk = () => device.createBuffer({ size: 8 * size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const a = mk(), b = mk();
      const data = new Float32Array(2 * size);
      for (let i = 0; i < data.length; i += 2) data[i] = Math.sin(i * 0.001);
      device.queue.writeBuffer(a, 0, data);
      const one = async () => {
        const t = performance.now();
        fft.forward(a, b); fft.inverse(b, a);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - t;
      };
      out.cold = await one();
      out.ms = out.cold > 1000 ? out.cold : await one();
      fft.destroy(); a.destroy(); b.destroy();
    })(), 2000, 'GPU micro-benchmark');
  } catch (e) {
    out.error = (e && e.message) || String(e);
    out.ms = Infinity;
  }
  out.total = performance.now() - t0;
  if (flags.gpuslow) { out.ms = 500; out.simulated = true; }
  return out;
}

// ---------------------------------------------------------------- 3D GPU memory estimate

// raw (unmerged) nLPT terms and parent terms whose gradient tensor sets are alive at the build peak, by order
const TERMS = { 1: [1, 0], 2: [2, 1], 3: [5, 3], 4: [13, 5] };

/**
 * Peak GPU memory of the 3D compute path at grid n and order: ~20 n^3 floats of working buffers, 3 n^3 floats per term,
 * 36 n^3 bytes per gradient tensor set during the build (+ 56 n^3 bytes FFT / complex work buffers).
 * @returns {{bytes:number, maxBuffer:number, terms:number}}
 */
export function gpu3dMemory(n, order) {
  const N = n * n * n, [terms, parents] = TERMS[Math.max(1, Math.min(4, order))];
  const bytes = 20 * 4 * N + 3 * 4 * N * terms + 36 * N * parents + 56 * N;
  const maxBuffer = Math.max(order >= 2 ? 36 * N : 0, 12 * N, 8 * N);
  return { bytes, maxBuffer, terms };
}

/**
 * Check whether the 3D GPU pipeline may be created. Returns null when fine or a message (fall back to WASM).
 * Refuses when the estimate exceeds 0.9 * min(maxBufferSize * 8, 2 GiB), when a single buffer exceeds
 * maxStorageBufferBindingSize / maxBufferSize, and (lite) above 64^3.
 */
export function gpu3dRefusal(device, n, order, lite) {
  const lim = device.limits;
  const mem = gpu3dMemory(n, order);
  if (lite && n > 64) return `lite mode caps the GPU 3D path at 64³ (grid ${n}³); using WASM`;
  const budget = 0.9 * Math.min(lim.maxBufferSize * 8, 2 * GiB);
  if (mem.bytes > budget) return `${n}³ with order ${order} needs about ${(mem.bytes / GiB).toFixed(2)} GiB of GPU memory, above the safe budget of ${(budget / GiB).toFixed(2)} GiB; using WASM`;
  const bind = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  if (mem.maxBuffer > bind) return `a ${(mem.maxBuffer / MiB).toFixed(0)} MiB buffer at ${n}³ exceeds the adapter limit maxStorageBufferBindingSize = ${(lim.maxStorageBufferBindingSize / MiB).toFixed(0)} MiB; using WASM`;
  return null;
}

// ---------------------------------------------------------------- diagnostics text

const fmtMiB = (v) => (v >= MiB ? `${(v / MiB).toFixed(0)} MiB` : String(v));
const LIMIT_KEYS = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension', 'maxStorageBuffersPerShaderStage', 'maxTextureDimension2D', 'maxTextureDimension3D', 'maxBindGroups'];

/** Plain-text diagnostics for pasting into a bug report. `d` = {gpu: gpuStatus(), bench, decision, backend, lite, pref, wasmBytes, lastError, flags, startupMs}. */
export function diagnosticsText(d) {
  const g = d.gpu || {}, i = g.info || {}, h = hostFacts();
  const L = [];
  L.push('Hopf-Cole cosmology lab diagnostics');
  L.push(`time: ${new Date().toISOString()}`);
  L.push(`url: ${location.href}`);
  L.push(`backend: ${d.backend}`);
  L.push(`WebGPU: ${g.available ? 'available' : 'unavailable (' + (g.reason || '?') + ')'}`);
  L.push(`adapter vendor: ${i.vendor || '-'}`);
  L.push(`adapter architecture: ${i.architecture || '-'}`);
  L.push(`adapter device: ${i.device || '-'}`);
  L.push(`adapter description: ${i.description || '-'}`);
  L.push(`isFallbackAdapter: ${i.isFallbackAdapter === undefined ? '-' : i.isFallbackAdapter}`);
  const lim = g.deviceLimits || g.limits;
  if (lim) for (const k of LIMIT_KEYS) if (lim[k] !== undefined) L.push(`limit ${k}: ${fmtMiB(lim[k])}${g.limits && g.limits[k] !== lim[k] ? ` (adapter max ${fmtMiB(g.limits[k])})` : ''}`);
  if (g.probeMs) L.push(`GPU probe time: ${g.probeMs.toFixed(0)} ms`);
  L.push(`hardwareConcurrency: ${h.hardwareConcurrency === undefined ? '-' : h.hardwareConcurrency}`);
  L.push(`deviceMemory: ${h.deviceMemory === undefined ? '- (not reported)' : h.deviceMemory + ' GB'}`);
  const b = d.bench;
  L.push(`micro-benchmark (64³ FFT fwd+inv): ${!b ? 'not run' : (b.error ? 'failed: ' + b.error : `${b.ms.toFixed(1)} ms${b.simulated ? ' (simulated by ?gpuslow)' : ''} (cold ${b.cold.toFixed(1)} ms, shader init ${b.init.toFixed(0)} ms)`)}`);
  L.push(`WASM memory: ${d.wasmBytes ? (d.wasmBytes / MiB).toFixed(0) + ' MiB' : 'unknown'}`);
  L.push(`performance preset: ${d.lite ? 'lite' : 'full'} (setting: ${d.pref}; ${d.decision ? d.decision.reason : '-'})`);
  if (d.startupMs) L.push(`start-up: ${d.startupMs}`);
  L.push(`last error: ${d.lastError || (g.lastError ? 'GPU: ' + g.lastError : 'none')}`);
  L.push(`user agent: ${h.userAgent}`);
  return L.join('\n');
}
