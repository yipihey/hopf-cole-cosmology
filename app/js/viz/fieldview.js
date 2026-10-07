// fieldview.js - scalar 2D field viewer.
//
//   const v = await new FieldView(canvas, {cmap:'viridis'}).init();
//   v.setField(float32Array, n /*rows (y)*/, m /*cols (x)*/);
//   v.setRange(undefined, undefined, {log:true});   // auto 0.5..99.5 percentile
//   v.draw();
//
// WebGPU path: full-screen triangle, r32float texture read with textureLoad
// (nearest, or manual bilinear), 256-entry LUT texture, uniform buffer with the
// range mapping. Canvas2D path: ImageData + drawImage.
// Row 0 of the data is drawn at the BOTTOM of the image; columns run along x.

import { cmapLUT } from './colormaps.js';
import {
  getGPU, compileModule, validated, surfaceMessage, observeCanvas, createOverlay,
  resolveRange, mapValue, pinCssSize,
} from './gpu.js';

const WGSL = /* wgsl */`
struct U {
  rect : vec4f,   // draw rectangle in canvas pixels: x0, y0, w, h
  vmin : f32,     // lower bound in mapping space (log10 if flags&1)
  vmax : f32,
  flags: u32,     // bit0: log, bit1: bilinear
  nx   : u32,     // columns
  ny   : u32,     // rows
  p0   : u32, p1: u32, p2: u32,
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var field : texture_2d<f32>;
@group(0) @binding(2) var lut   : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  // one big triangle covering the viewport
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn fetchF(ix : i32, iy : i32) -> f32 {
  let c = vec2<i32>(clamp(ix, 0, i32(u.nx) - 1), clamp(iy, 0, i32(u.ny) - 1));
  return textureLoad(field, c, 0).x;
}

fn lutColor(t : f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(x));
  let i1 = min(i0 + 1, 255);
  let f = x - floor(x);
  return mix(textureLoad(lut, vec2<i32>(i0, 0), 0).rgb, textureLoad(lut, vec2<i32>(i1, 0), 0).rgb, f);
}

@fragment
fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f {
  let uv = (pos.xy - u.rect.xy) / u.rect.zw;           // 0..1, y down
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0) { return vec4f(0.0); }
  // data row 0 is at the bottom of the image
  let cx = uv.x * f32(u.nx);
  let cy = (1.0 - uv.y) * f32(u.ny);
  var v : f32;
  if ((u.flags & 2u) != 0u) {
    let gx = cx - 0.5; let gy = cy - 0.5;
    let x0 = floor(gx); let y0 = floor(gy);
    let fx = gx - x0; let fy = gy - y0;
    let ix = i32(x0); let iy = i32(y0);
    v = mix(mix(fetchF(ix, iy), fetchF(ix + 1, iy), fx), mix(fetchF(ix, iy + 1), fetchF(ix + 1, iy + 1), fx), fy);
  } else {
    v = fetchF(i32(floor(cx)), i32(floor(cy)));
  }
  if ((u.flags & 1u) != 0u) { v = log(max(v, 1e-30)) * 0.4342944819; }
  let t = (v - u.vmin) / (u.vmax - u.vmin);
  return vec4f(lutColor(t), 1.0);
}
`;

export class FieldView {
  /**
   * @param {HTMLCanvasElement} canvas  should have a CSS size (e.g. width:100%; aspect-ratio:1)
   * @param {{cmap?:string, interpolate?:boolean, fit?:'contain'|'stretch'}} [opts]
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = { cmap: 'viridis', interpolate: false, fit: 'contain', ...opts };
    this.cmap = this.opts.cmap;
    this.interpolate = !!this.opts.interpolate;
    this.backend = null;           // 'webgpu' | 'canvas2d' after init()
    this.data = null; this.n = 0; this.m = 0;
    this.viewRect = { x: 0, y: 0, w: 0, h: 0 };   // image rectangle in canvas pixels (valid after draw)
    this._req = { vmin: undefined, vmax: undefined, log: false, symmetric: false };
    this.range = { vmin: 0, vmax: 1, log: false };
    this._overlayFn = null; this._overlay = null;
    this._initPromise = null; this._dirty = true;
  }

  /** Acquire WebGPU if possible, else Canvas2D. Safe to call repeatedly. @returns {Promise<FieldView>} */
  init() {
    if (!this._initPromise) this._initPromise = this._init().then(() => this);
    return this._initPromise;
  }

  async _init() {
    const canvas = this.canvas;
    pinCssSize(canvas);
    const g = await getGPU();
    if (g) {
      try { await this._initGPU(g); this.backend = 'webgpu'; }
      catch (e) { console.error('[viz] FieldView WebGPU init failed, falling back to Canvas2D:', e); this._gpu = null; }
    }
    if (!this.backend) {
      this.ctx2d = canvas.getContext('2d');
      if (!this.ctx2d) { surfaceMessage(canvas, 'FieldView: could not obtain a rendering context.'); throw new Error('no rendering context'); }
      this.backend = 'canvas2d';
      this._off = document.createElement('canvas');
    }
    this._ro = observeCanvas(canvas, (changed) => { if (changed && this.data) { this._dirty = true; this.draw(); } });
  }

  async _initGPU({ device, format }) {
    const module = await compileModule(device, WGSL, 'FieldView');
    const { pipeline, layout } = await validated(device, async () => {
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ] });
      const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      return { pipeline, layout };
    });
    const ubuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const lutTex = device.createTexture({ size: [256, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    // Acquire the canvas context last: once it is 'webgpu' the canvas cannot become '2d'.
    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('canvas.getContext("webgpu") returned null');
    context.configure({ device, format, alphaMode: 'premultiplied' });
    this._gpu = { device, format, module, pipeline, layout, ubuf, lutTex, context, fieldTex: null, bind: null };
    this._ubAB = new ArrayBuffer(48);
    this._ubF = new Float32Array(this._ubAB); this._ubU = new Uint32Array(this._ubAB);
    this._uploadLUT();
  }

  // -- state setters --------------------------------------------------------

  /**
   * Upload a field. data[r*m + c]: r = row (y, row 0 at the bottom), c = column (x).
   * @param {Float32Array} data length n*m
   */
  setField(data, n, m) {
    if (!(n > 0 && m > 0) || data.length < n * m) throw new Error(`FieldView.setField: data length ${data.length} < n*m = ${n * m}`);
    if (!(data instanceof Float32Array)) data = Float32Array.from(data);
    this.data = data; this.n = n; this.m = m;
    this._resolveRange();
    this._dirty = true;
    const G = this._gpu;
    if (G) {
      const { device } = G;
      const maxD = device.limits.maxTextureDimension2D;
      if (n > maxD || m > maxD) { surfaceMessage(this.canvas, `FieldView: field ${m}x${n} exceeds GPU texture limit ${maxD}.`); throw new Error('field too large'); }
      if (!G.fieldTex || G.fieldTex.width !== m || G.fieldTex.height !== n) {
        if (G.fieldTex) G.fieldTex.destroy();
        G.fieldTex = device.createTexture({ size: [m, n], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        G.bind = null;
      }
      device.queue.writeTexture({ texture: G.fieldTex }, data, { bytesPerRow: m * 4, rowsPerImage: n }, { width: m, height: n });
    }
  }

  /**
   * Colour mapping. vmin/vmax in data units; undefined -> auto (0.5..99.5 percentile).
   * log: map log10(max(v, tiny)). symmetric: force [-a, a] (ignored for log).
   */
  setRange(vmin, vmax, { log = false, symmetric = false } = {}) {
    this._req = { vmin, vmax, log, symmetric };
    this._resolveRange();
    this._dirty = true;
  }

  setColormap(name) {
    this.cmap = name; this._dirty = true;
    if (this._gpu) this._uploadLUT();
  }

  setInterpolate(on) { this.interpolate = !!on; this._dirty = true; }

  /** Register a Canvas2D overlay: fn(ctx, widthPx, heightPx); this.viewRect gives the image rect. */
  setOverlay(fn) {
    this._overlayFn = fn;
    if (fn && !this._overlay) this._overlay = createOverlay(this.canvas);
    if (!fn && this._overlay) this._overlay.ctx.clearRect(0, 0, this._overlay.canvas.width, this._overlay.canvas.height);
  }

  /** Current effective range {vmin, vmax, log} (data units) - use for colorbars. */
  getRange() { return { ...this.range }; }

  /** Map a mouse event to {row, col, value} (or null outside the image). */
  pick(ev) {
    if (!this.data) return null;
    const r = this.canvas.getBoundingClientRect();
    const px = (ev.clientX - r.left) * this.canvas.width / r.width, py = (ev.clientY - r.top) * this.canvas.height / r.height;
    const R = this.viewRect;
    const u = (px - R.x) / R.w, v = (py - R.y) / R.h;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    const col = Math.floor(u * this.m), row = Math.floor((1 - v) * this.n);
    return { row, col, value: this.data[row * this.m + col] };
  }

  destroy() {
    if (this._ro) this._ro.disconnect();
    if (this._overlay) this._overlay.remove();
    if (this._gpu && this._gpu.fieldTex) this._gpu.fieldTex.destroy();
  }

  // -- internals ------------------------------------------------------------

  _resolveRange() { this.range = resolveRange(this._req, this.data); }

  _uploadLUT() {
    const G = this._gpu;
    G.device.queue.writeTexture({ texture: G.lutTex }, cmapLUT(this.cmap, 256), { bytesPerRow: 1024 }, { width: 256, height: 1 });
  }

  _computeRect() {
    const W = this.canvas.width, H = this.canvas.height;
    let w = W, h = H;
    if (this.opts.fit !== 'stretch') {
      const s = Math.min(W / this.m, H / this.n);
      w = this.m * s; h = this.n * s;
    }
    this.viewRect = { x: Math.round((W - w) / 2), y: Math.round((H - h) / 2), w: Math.round(w), h: Math.round(h) };
  }

  /** Render the current field. No-op until init() and setField() have happened. */
  draw() {
    if (!this.backend || !this.data) return;
    try {
      this._computeRect();
      if (this.backend === 'webgpu') this._drawGPU(); else this._draw2D();
      this._drawOverlay();
    } catch (e) {
      surfaceMessage(this.canvas, 'FieldView draw failed: ' + e.message);
    }
  }

  _drawOverlay() {
    const o = this._overlay;
    if (!o) return;
    o.sync();
    o.ctx.setTransform(1, 0, 0, 1, 0, 0);
    o.ctx.clearRect(0, 0, o.canvas.width, o.canvas.height);
    if (this._overlayFn) { o.ctx.save(); this._overlayFn(o.ctx, o.canvas.width, o.canvas.height); o.ctx.restore(); }
  }

  _drawGPU() {
    const G = this._gpu, { device } = G, R = this.viewRect, rg = this.range;
    if (!G.bind) {
      G.bind = device.createBindGroup({ layout: G.layout, entries: [
        { binding: 0, resource: { buffer: G.ubuf } },
        { binding: 1, resource: G.fieldTex.createView() },
        { binding: 2, resource: G.lutTex.createView() },
      ] });
    }
    const f = this._ubF, u = this._ubU;
    f[0] = R.x; f[1] = R.y; f[2] = R.w; f[3] = R.h;
    f[4] = rg.log ? Math.log10(rg.vmin) : rg.vmin;
    f[5] = rg.log ? Math.log10(rg.vmax) : rg.vmax;
    u[6] = (rg.log ? 1 : 0) | (this.interpolate ? 2 : 0);
    u[7] = this.m; u[8] = this.n;
    device.queue.writeBuffer(G.ubuf, 0, this._ubAB);
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: G.context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(G.pipeline);
    pass.setBindGroup(0, G.bind);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  _draw2D() {
    const ctx = this.ctx2d, { n, m } = this, R = this.viewRect;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const off = this._off;
    if (this._dirty || off.width !== m || off.height !== n) {
      off.width = m; off.height = n;
      const octx = off.getContext('2d');
      const img = octx.createImageData(m, n), px = img.data, lut = cmapLUT(this.cmap, 256), d = this.data, rg = this.range;
      for (let r = 0; r < n; r++) {
        const yrow = (n - 1 - r) * m;           // row 0 at the bottom
        for (let c = 0; c < m; c++) {
          const v = d[r * m + c], o = 4 * (yrow + c);
          if (v !== v) { px[o + 3] = 0; continue; } // NaN -> transparent
          const k = 4 * Math.round(mapValue(v, rg) * 255);
          px[o] = lut[k]; px[o + 1] = lut[k + 1]; px[o + 2] = lut[k + 2]; px[o + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      this._dirty = false;
    }
    ctx.imageSmoothingEnabled = this.interpolate;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, R.x, R.y, R.w, R.h);
  }
}
