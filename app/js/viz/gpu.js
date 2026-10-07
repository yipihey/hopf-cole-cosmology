// gpu.js - shared WebGPU / canvas plumbing for the hcc visualization library.
//
//  * One shared GPUDevice for all views (getGPU()).
//  * `?nogpu=1` in the page URL forces the Canvas2D fallbacks.
//  * Helpers for canvas sizing (devicePixelRatio aware), overlay canvases,
//    user-visible error messages, and robust-percentile auto ranging.

/** True when the page URL contains `nogpu` (used to test the fallback paths). */
export function gpuDisabled() {
  try { return new URLSearchParams(location.search).has('nogpu'); } catch (e) { return false; }
}

let _gpuPromise = null;

/**
 * Resolve to {adapter, device, format} or null when WebGPU is unavailable /
 * disabled / the adapter or device request fails. Memoized.
 */
export function getGPU() {
  if (_gpuPromise) return _gpuPromise;
  _gpuPromise = (async () => {
    if (gpuDisabled()) { console.info('[viz] WebGPU disabled by ?nogpu'); return null; }
    if (typeof navigator === 'undefined' || !navigator.gpu) { console.info('[viz] navigator.gpu missing; using Canvas2D'); return null; }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { console.info('[viz] no WebGPU adapter; using Canvas2D'); return null; }
      const device = await adapter.requestDevice();
      device.lost.then((info) => { console.warn('[viz] WebGPU device lost:', info.message); _gpuPromise = null; });
      device.addEventListener('uncapturederror', (ev) => console.error('[viz] WebGPU error:', ev.error && ev.error.message));
      return { adapter, device, format: navigator.gpu.getPreferredCanvasFormat() };
    } catch (e) {
      console.warn('[viz] WebGPU init failed; using Canvas2D:', e);
      return null;
    }
  })();
  return _gpuPromise;
}

/** Compile WGSL and throw (with line info) if there are compilation errors. */
export async function compileModule(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error');
  if (errs.length) {
    throw new Error(`WGSL compile error in ${label}: ` + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  }
  return module;
}

/** Run `fn` inside a validation error scope; rethrow any validation error. */
export async function validated(device, fn) {
  device.pushErrorScope('validation');
  let out;
  try { out = await fn(); } catch (e) { await device.popErrorScope(); throw e; }
  const err = await device.popErrorScope();
  if (err) throw new Error('WebGPU validation error: ' + err.message);
  return out;
}

// ---------------------------------------------------------------------------
// Visible messages

/** Show (or clear, when text is falsy) a message box next to the canvas. Also logs to console. */
export function surfaceMessage(canvas, text, isError = true) {
  if (text) (isError ? console.error : console.warn)('[viz]', text);
  const parent = canvas.parentElement;
  if (!parent) return;
  let box = canvas._hccMsg;
  if (!text) { if (box) box.remove(); canvas._hccMsg = null; return; }
  if (!box) {
    box = document.createElement('div');
    box.className = 'hcc-viz-msg';
    box.setAttribute('role', 'alert');
    box.style.cssText = 'margin:4px 0;padding:6px 8px;border:1px solid #d55e00;border-radius:4px;' +
      'font:12px/1.4 system-ui,sans-serif;color:#d55e00;background:rgba(213,94,0,.08);white-space:pre-wrap';
    canvas.after(box);
    canvas._hccMsg = box;
  }
  box.textContent = text;
}

// ---------------------------------------------------------------------------
// Canvas sizing

/**
 * Canvases whose CSS size comes from their width/height *attributes* would grow
 * without bound when we resize the backing store for devicePixelRatio. Detect
 * that case and pin the CSS size explicitly. Idempotent.
 */
export function pinCssSize(canvas) {
  if (canvas._hccPinned) return;
  const w0 = canvas.width, h0 = canvas.height;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return; // hidden: cannot tell; try again later
  canvas._hccPinned = true;
  canvas.width = w0 + 7; canvas.height = h0 + 7;
  const intrinsicW = canvas.clientWidth !== cw, intrinsicH = canvas.clientHeight !== ch;
  canvas.width = w0; canvas.height = h0;
  if (intrinsicW && !canvas.style.width) canvas.style.width = cw + 'px';
  if (intrinsicH && !canvas.style.height) canvas.style.height = ch + 'px';
}

/** Size the backing store to CSS size * devicePixelRatio. Returns true if it changed. */
export function fitCanvas(canvas, maxDim = 8192) {
  pinCssSize(canvas);
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return false;
  const w = Math.min(maxDim, Math.max(1, Math.round(cw * dpr)));
  const h = Math.min(maxDim, Math.max(1, Math.round(ch * dpr)));
  if (canvas.width === w && canvas.height === h) return false;
  canvas.width = w; canvas.height = h;
  return true;
}

/**
 * Keep the canvas backing store in sync with its CSS box (ResizeObserver +
 * devicePixelRatio changes). `cb(changed)` is called after each resize.
 */
export function observeCanvas(canvas, cb, maxDim = 8192) {
  const run = () => { const ch = fitCanvas(canvas, maxDim); cb(ch); };
  let ro = null;
  if (typeof ResizeObserver !== 'undefined') { ro = new ResizeObserver(run); ro.observe(canvas); }
  let mq = null;
  const onDpr = () => { run(); watchDpr(); };
  const watchDpr = () => {
    if (mq) mq.removeEventListener('change', onDpr);
    mq = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    mq.addEventListener('change', onDpr, { once: true });
  };
  watchDpr();
  run();
  return { disconnect() { if (ro) ro.disconnect(); if (mq) mq.removeEventListener('change', onDpr); } };
}

// ---------------------------------------------------------------------------
// Overlay canvas (transparent Canvas2D layered over the main canvas)

export function createOverlay(canvas) {
  const parent = canvas.parentElement;
  if (parent && getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
  const c = document.createElement('canvas');
  c.className = 'hcc-overlay';
  c.style.cssText = 'position:absolute;pointer-events:none;';
  canvas.after(c);
  const ctx = c.getContext('2d');
  const sync = () => {
    c.style.left = canvas.offsetLeft + 'px'; c.style.top = canvas.offsetTop + 'px';
    c.style.width = canvas.offsetWidth + 'px'; c.style.height = canvas.offsetHeight + 'px';
    if (c.width !== canvas.width || c.height !== canvas.height) { c.width = canvas.width; c.height = canvas.height; }
  };
  sync();
  return { canvas: c, ctx, sync, remove() { c.remove(); } };
}

// ---------------------------------------------------------------------------
// Range helpers

/**
 * Robust [lo, hi] percentile range of the finite (and, if positiveOnly, >0)
 * entries of `data`. Subsamples large arrays.
 * @param {ArrayLike<number>} data
 * @param {number} pLo percentile in [0,100]
 * @param {number} pHi percentile in [0,100]
 */
export function percentileRange(data, pLo = 0.5, pHi = 99.5, positiveOnly = false) {
  const N = data.length;
  const stride = Math.max(1, Math.floor(N / 200000));
  const buf = new Float32Array(Math.ceil(N / stride));
  let k = 0;
  for (let i = 0; i < N; i += stride) {
    const v = data[i];
    if (Number.isFinite(v) && (!positiveOnly || v > 0)) buf[k++] = v;
  }
  if (!k) return [positiveOnly ? 1e-3 : 0, 1];
  const s = buf.subarray(0, k).sort();
  const at = (p) => s[Math.min(k - 1, Math.max(0, Math.round((p / 100) * (k - 1))))];
  let lo = at(pLo), hi = at(pHi);
  if (!(hi > lo)) { hi = lo + (Math.abs(lo) || 1) * 1e-3; }
  return [lo, hi];
}

/**
 * Resolve a user range request {vmin,vmax,log,symmetric} against data.
 * Returned bounds are always in data units; for log they are positive.
 */
export function resolveRange(req, data) {
  const log = !!req.log;
  let lo = req.vmin, hi = req.vmax;
  if ((lo === undefined || hi === undefined) && data) {
    const [alo, ahi] = percentileRange(data, 0.5, 99.5, log);
    if (lo === undefined) lo = alo;
    if (hi === undefined) hi = ahi;
  }
  if (lo === undefined) lo = log ? 1e-3 : 0;
  if (hi === undefined) hi = 1;
  if (req.symmetric && !log) { const a = Math.max(Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
  if (log) { lo = Math.max(lo, 1e-30); if (!(hi > lo)) hi = lo * 10; }
  else if (!(hi > lo)) hi = lo + 1e-6 * (Math.abs(lo) || 1);
  return { vmin: lo, vmax: hi, log };
}

/** Map v to [0,1] for the range returned by resolveRange (CPU side). */
export function mapValue(v, r) {
  if (r.log) {
    const a = Math.log10(r.vmin), b = Math.log10(r.vmax);
    const x = Math.log10(Math.max(v, 1e-30));
    return Math.min(1, Math.max(0, (x - a) / (b - a)));
  }
  return Math.min(1, Math.max(0, (v - r.vmin) / (r.vmax - r.vmin)));
}
