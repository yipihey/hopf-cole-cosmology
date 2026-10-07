// volumeview.js - 3D scalar volume viewer.
//
//   const v = await new VolumeView(canvas, {cmap:'magma'}).init();
//   v.setVolume(float32Array /* n^3, index = (i*n + j)*n + k */, n);
//   v.setRange(undefined, undefined, {log:true});
//   v.draw({mode:'emission', opacity: 8});
//
// Axes: i = x, j = y, k = z; the volume fills the cube [-0.5,0.5]^3 and the
// camera orbits it with z up. Drag = rotate, wheel = zoom, double-click = reset.
//
// WebGPU: fragment-shader ray marching through an r32float 3D texture with
// manual trilinear interpolation (textureLoad, no filtering needed) and the
// colormap LUT as transfer function. Modes: 'mip', 'emission', 'slice'.
// Canvas2D fallback: the three central slices side by side (or one slice).

import { cmapLUT } from './colormaps.js';
import {
  getGPU, compileModule, validated, surfaceMessage, observeCanvas, resolveRange, mapValue, pinCssSize,
} from './gpu.js';

const WGSL = /* wgsl */`
struct U {
  eye   : vec4f,  // xyz eye position; w unused
  right : vec4f,  // xyz camera right; w = aspect (W/H)
  up    : vec4f,  // xyz camera up;    w = tan(fov/2) (perspective) or half-height (orthographic)
  fwd   : vec4f,  // xyz view direction; w = 1 perspective, 0 orthographic
  rng   : vec4f,  // lo, hi (mapping space), log flag, opacity (optical depth per unit length at t=1)
  geo   : vec4f,  // n, steps, slice axis, slice index
  misc  : vec4f,  // mode (0 mip, 1 emission, 2 slice), box-line alpha, 0, 0
  bg    : vec4f,  // background rgb, 0
  rect  : vec4f,  // ray modes: 0,0,W,H ; slice mode: square draw rect x0,y0,size,size (canvas px)
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var vol : texture_3d<f32>;
@group(0) @binding(2) var lut : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn mapT(v : f32) -> f32 {
  var x = v;
  if (u.rng.z > 0.5) { x = log(max(v, 1e-30)) * 0.4342944819; }
  return clamp((x - u.rng.x) / (u.rng.y - u.rng.x), 0.0, 1.0);
}

fn lutColor(t : f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(x));
  let i1 = min(i0 + 1, 255);
  return mix(textureLoad(lut, vec2<i32>(i0, 0), 0).rgb, textureLoad(lut, vec2<i32>(i1, 0), 0).rgb, x - floor(x));
}

// data index (ix,iy,iz) -> value; periodic. texture axes: x = k (z), y = j (y), z = i (x)
fn texel(i : vec3<i32>) -> f32 {
  let n = vec3<i32>(i32(u.geo.x));
  let w = (i + n) % n;
  return textureLoad(vol, vec3<i32>(w.z, w.y, w.x), 0).x;
}

// trilinear interpolation at box position p in [-0.5,0.5]^3
fn sampleVol(p : vec3f) -> f32 {
  let g = (p + vec3f(0.5)) * u.geo.x - vec3f(0.5);
  let f0 = floor(g);
  let w = g - f0;
  let i = vec3<i32>(f0);
  let c00 = mix(texel(i), texel(i + vec3<i32>(1, 0, 0)), w.x);
  let c10 = mix(texel(i + vec3<i32>(0, 1, 0)), texel(i + vec3<i32>(1, 1, 0)), w.x);
  let c01 = mix(texel(i + vec3<i32>(0, 0, 1)), texel(i + vec3<i32>(1, 0, 1)), w.x);
  let c11 = mix(texel(i + vec3<i32>(0, 1, 1)), texel(i + vec3<i32>(1, 1, 1)), w.x);
  return mix(mix(c00, c10, w.y), mix(c01, c11, w.y), w.z);
}

fn hash12(p : vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn boxHit(o : vec3f, d0 : vec3f) -> vec2f {
  let d = select(d0, vec3f(1e-8), abs(d0) < vec3f(1e-8));
  let inv = vec3f(1.0) / d;
  let t0 = (vec3f(-0.5) - o) * inv;
  let t1 = (vec3f(0.5) - o) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// 1 when p lies close to a cube edge (two coordinates near +-0.5)
fn edgeMask(p : vec3f) -> f32 {
  let a = abs(abs(p) - vec3f(0.5));
  let e = 0.004;
  let c = select(0.0, 1.0, a.x < e) + select(0.0, 1.0, a.y < e) + select(0.0, 1.0, a.z < e);
  return select(0.0, 1.0, c >= 2.0);
}

@fragment
fn fs(@builtin(position) fp : vec4f) -> @location(0) vec4f {
  let mode = i32(u.misc.x);
  let n = u.geo.x;

  if (mode == 2) {
    // ---- single axis-aligned slice, nearest sampling (FieldView-like) ----
    let q = (fp.xy - u.rect.xy) / u.rect.z;
    if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { return vec4f(0.0); }
    let a = vec2<i32>(floor(vec2f(q.x, 1.0 - q.y) * n));
    let idx = i32(u.geo.w);
    var c = vec3<i32>(a.x, a.y, idx);
    let axis = i32(u.geo.z);
    if (axis == 0) { c = vec3<i32>(idx, a.x, a.y); }
    else if (axis == 1) { c = vec3<i32>(a.x, idx, a.y); }
    return vec4f(lutColor(mapT(texel(c))), 1.0);
  }

  // ---- camera ray ----
  let ndc = vec2f(fp.x / u.rect.z * 2.0 - 1.0, 1.0 - fp.y / u.rect.w * 2.0);
  var ro = u.eye.xyz;
  var rd = u.fwd.xyz;
  if (u.fwd.w > 0.5) {
    rd = normalize(u.fwd.xyz + (ndc.x * u.right.w * u.up.w) * u.right.xyz + (ndc.y * u.up.w) * u.up.xyz);
  } else {
    ro = ro + (ndc.x * u.right.w * u.up.w) * u.right.xyz + (ndc.y * u.up.w) * u.up.xyz;
  }
  let hit = boxHit(ro, rd);
  let tn = max(hit.x, 0.0);
  let tf = hit.y;
  var col = u.bg.rgb;
  if (tf > tn) {
    let steps = i32(u.geo.y);
    let dt = (tf - tn) / f32(steps);
    var t = tn + dt * hash12(fp.xy);             // jittered start hides banding
    if (mode == 0) {
      var m = 0.0;
      for (var s = 0; s < steps; s = s + 1) {
        m = max(m, mapT(sampleVol(ro + rd * t)));
        t = t + dt;
      }
      col = mix(u.bg.rgb, lutColor(m), smoothstep(0.0, 0.06, m));
    } else {
      var C = vec3f(0.0);
      var A = 0.0;
      for (var s = 0; s < steps; s = s + 1) {
        let v = mapT(sampleVol(ro + rd * t));
        let a = 1.0 - exp(-u.rng.w * v * v * dt);
        C = C + (1.0 - A) * a * lutColor(v);
        A = A + (1.0 - A) * a;
        if (A > 0.985) { break; }
        t = t + dt;
      }
      col = C + (1.0 - A) * u.bg.rgb;
    }
    // faint bounding-box edges (front and back)
    let ea = u.misc.y;
    if (ea > 0.0) {
      col = mix(col, vec3f(1.0), ea * edgeMask(ro + rd * tn));
      col = mix(col, vec3f(1.0), 0.5 * ea * edgeMask(ro + rd * tf));
    }
  }
  return vec4f(col, 1.0);
}
`;

const DEFAULTS = { mode: 'mip', axis: 2, index: undefined, opacity: 8, steps: undefined, boxLines: 0.35 };

export class VolumeView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{cmap?:string, projection?:'perspective'|'orthographic', interactive?:boolean, background?:number[]}} [opts]
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = { cmap: 'viridis', projection: 'perspective', interactive: true, background: [0.04, 0.045, 0.06], ...opts };
    this.cmap = this.opts.cmap;
    this.backend = null;
    this.data = null; this.n = 0;
    this._req = { vmin: undefined, vmax: undefined, log: false };
    this.range = { vmin: 0, vmax: 1, log: false };
    this.cam = { azimuth: 0.75, elevation: 0.45, zoom: 1, ortho: this.opts.projection === 'orthographic' };
    this.params = { ...DEFAULTS };
    this._initPromise = null; this._raf = 0;
  }

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
      catch (e) { console.error('[viz] VolumeView WebGPU init failed, falling back to Canvas2D:', e); this._gpu = null; }
    }
    if (!this.backend) {
      this.ctx2d = canvas.getContext('2d');
      if (!this.ctx2d) { surfaceMessage(canvas, 'VolumeView: could not obtain a rendering context.'); throw new Error('no rendering context'); }
      this.backend = 'canvas2d';
      this._off = document.createElement('canvas');
    }
    this._ro = observeCanvas(canvas, (changed) => { if (changed && this.data) this.draw(); });
    if (this.opts.interactive && this.backend === 'webgpu') this._bindInteraction();
  }

  async _initGPU({ device, format }) {
    const module = await compileModule(device, WGSL, 'VolumeView');
    const { pipeline, layout } = await validated(device, async () => {
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
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
    const ubuf = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const lutTex = device.createTexture({ size: [256, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('canvas.getContext("webgpu") returned null');
    context.configure({ device, format, alphaMode: 'premultiplied' });
    this._gpu = { device, format, pipeline, layout, ubuf, lutTex, context, volTex: null, bind: null, lutName: null };
    this._ub = new Float32Array(36);
  }

  // -- state ----------------------------------------------------------------

  /** @param {Float32Array} data n^3 values, index = (i*n + j)*n + k  @param {number} n */
  setVolume(data, n) {
    if (data.length < n * n * n) throw new Error(`VolumeView.setVolume: need ${n ** 3} values, got ${data.length}`);
    if (!(data instanceof Float32Array)) data = Float32Array.from(data);
    const resized = n !== this.n;
    this.data = data; this.n = n;
    this._resolveRange();
    const G = this._gpu;
    if (G) {
      const { device } = G;
      const maxD = device.limits.maxTextureDimension3D;
      if (n > maxD) { surfaceMessage(this.canvas, `VolumeView: n=${n} exceeds the 3D texture limit ${maxD}.`); throw new Error('volume too large'); }
      if (!G.volTex || resized) {
        if (G.volTex) G.volTex.destroy();
        G.volTex = device.createTexture({ size: [n, n, n], dimension: '3d', format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        G.bind = null;
      }
      // texture x = k (fastest), y = j, z = i  ==  memory order of the data array
      device.queue.writeTexture({ texture: G.volTex }, data, { bytesPerRow: n * 4, rowsPerImage: n }, { width: n, height: n, depthOrArrayLayers: n });
    }
  }

  /** Colour range in data units; undefined -> 0.5..99.5 percentile. log: map log10. */
  setRange(vmin, vmax, { log = false } = {}) {
    this._req = { vmin, vmax, log };
    this._resolveRange();
  }

  setColormap(name) { this.cmap = name; }

  /** Select an axis-aligned slice (axis 0=x(i), 1=y(j), 2=z(k)) and switch to 'slice' mode. */
  setSlice(axis, index) { Object.assign(this.params, { mode: 'slice', axis, index }); }

  /** Set camera angles (radians), zoom and/or projection. */
  setCamera({ azimuth, elevation, zoom, projection } = {}) {
    const c = this.cam;
    if (azimuth !== undefined) c.azimuth = azimuth;
    if (elevation !== undefined) c.elevation = Math.max(-1.55, Math.min(1.55, elevation));
    if (zoom !== undefined) c.zoom = Math.max(0.3, Math.min(10, zoom));
    if (projection !== undefined) c.ortho = projection === 'orthographic';
  }

  getRange() { return { ...this.range }; }

  destroy() {
    if (this._ro) this._ro.disconnect();
    if (this._gpu && this._gpu.volTex) this._gpu.volTex.destroy();
    if (this._unbind) this._unbind();
  }

  _resolveRange() { this.range = resolveRange(this._req, this.data); }

  // -- interaction ----------------------------------------------------------

  _bindInteraction() {
    const cv = this.canvas;
    cv.style.touchAction = 'none';
    cv.style.cursor = 'grab';
    let drag = null;
    const down = (e) => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; };
    const move = (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      this.setCamera({ azimuth: this.cam.azimuth - dx * 0.008, elevation: this.cam.elevation + dy * 0.008 });
      this._schedule();
    };
    const up = (e) => { drag = null; cv.style.cursor = 'grab'; try { cv.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ } };
    const wheel = (e) => {
      e.preventDefault();
      this.setCamera({ zoom: this.cam.zoom * Math.exp(-e.deltaY * 0.0012) });
      this._schedule();
    };
    const dbl = () => { this.setCamera({ azimuth: 0.75, elevation: 0.45, zoom: 1 }); this._schedule(); };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('wheel', wheel, { passive: false }); cv.addEventListener('dblclick', dbl);
    this._unbind = () => {
      cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move);
      cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up);
      cv.removeEventListener('wheel', wheel); cv.removeEventListener('dblclick', dbl);
    };
  }

  _schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }

  // -- drawing --------------------------------------------------------------

  /**
   * @param {{mode?:'mip'|'emission'|'slice', axis?:0|1|2, index?:number, opacity?:number, steps?:number, boxLines?:number}} [p]
   */
  draw(p = {}) {
    if (!this.backend || !this.data) return;
    Object.assign(this.params, p);
    try {
      if (this.backend === 'webgpu') this._drawGPU(); else this._draw2D();
    } catch (e) {
      surfaceMessage(this.canvas, 'VolumeView draw failed: ' + e.message);
    }
  }

  _camera() {
    const c = this.cam, ce = Math.cos(c.elevation), se = Math.sin(c.elevation);
    const dir = [ce * Math.cos(c.azimuth), ce * Math.sin(c.azimuth), se];       // from target to eye
    const dist = c.ortho ? 3 : 3.7 / c.zoom;
    const eye = dir.map((x) => x * dist);
    const fwd = dir.map((x) => -x);
    // right = fwd x zhat, up = right x fwd
    let right = [fwd[1], -fwd[0], 0];
    const rl = Math.hypot(right[0], right[1]) || 1;
    right = right.map((x) => x / rl);
    const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
    return { eye, fwd, right, up, scale: c.ortho ? 0.95 / c.zoom : Math.tan(0.5 * 35 * Math.PI / 180) };
  }

  _drawGPU() {
    const G = this._gpu, { device } = G, P = this.params, n = this.n, rg = this.range;
    const W = this.canvas.width, H = this.canvas.height;
    if (!G.bind) {
      G.bind = device.createBindGroup({ layout: G.layout, entries: [
        { binding: 0, resource: { buffer: G.ubuf } },
        { binding: 1, resource: G.volTex.createView({ dimension: '3d' }) },
        { binding: 2, resource: G.lutTex.createView() },
      ] });
    }
    if (G.lutName !== this.cmap) {
      device.queue.writeTexture({ texture: G.lutTex }, cmapLUT(this.cmap, 256), { bytesPerRow: 1024 }, { width: 256, height: 1 });
      G.lutName = this.cmap;
    }
    const cam = this._camera(), u = this._ub, bg = this.opts.background;
    const mode = { mip: 0, emission: 1, slice: 2 }[P.mode];
    if (mode === undefined) throw new Error(`unknown mode "${P.mode}"`);
    const steps = P.steps || Math.min(512, Math.max(96, Math.round(1.6 * n)));
    const S = Math.min(W, H), idx = Math.max(0, Math.min(n - 1, P.index === undefined ? n >> 1 : Math.round(P.index)));
    const rect = mode === 2 ? [Math.floor((W - S) / 2), Math.floor((H - S) / 2), S, S] : [0, 0, W, H];
    u.set([
      ...cam.eye, 0,
      ...cam.right, W / H,
      ...cam.up, cam.scale,
      ...cam.fwd, this.cam.ortho ? 0 : 1,
      rg.log ? Math.log10(rg.vmin) : rg.vmin, rg.log ? Math.log10(rg.vmax) : rg.vmax, rg.log ? 1 : 0, P.opacity,
      n, steps, P.axis, idx,
      mode, P.boxLines, 0, 0,
      bg[0], bg[1], bg[2], 0,
      ...rect,
    ]);
    device.queue.writeBuffer(G.ubuf, 0, u);
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: G.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(G.pipeline);
    pass.setBindGroup(0, G.bind);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  /** CPU: extract slice `idx` perpendicular to `axis` as an n*n RGBA image (row 0 of the image = top). */
  _sliceImage(axis, idx, lut) {
    const { n, data: d, range: rg } = this;
    const off = this._off; off.width = n; off.height = n;
    const octx = off.getContext('2d'), img = octx.createImageData(n, n), px = img.data;
    for (let b = 0; b < n; b++) {
      const row = (n - 1 - b) * n;                    // vertical coordinate b points up
      for (let a = 0; a < n; a++) {
        const v = axis === 0 ? d[(idx * n + a) * n + b] : axis === 1 ? d[(a * n + idx) * n + b] : d[(a * n + b) * n + idx];
        const o = 4 * (row + a);
        if (v !== v) { px[o + 3] = 0; continue; }
        const k = 4 * Math.round(mapValue(v, rg) * 255);
        px[o] = lut[k]; px[o + 1] = lut[k + 1]; px[o + 2] = lut[k + 2]; px[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    return off;
  }

  _draw2D() {
    const ctx = this.ctx2d, P = this.params, n = this.n, W = this.canvas.width, H = this.canvas.height;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const lut = cmapLUT(this.cmap, 256);
    const color = getComputedStyle(this.canvas).color || '#888';
    const names = [['y', 'z', 'x'], ['x', 'z', 'y'], ['x', 'y', 'z']];   // [horizontal, vertical, fixed] per axis
    const single = P.mode === 'slice';
    const axes = single ? [P.axis] : [0, 1, 2];
    const cellW = W / axes.length, labelH = 18 * dpr;
    ctx.font = `${12 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = color;
    ctx.imageSmoothingEnabled = false;
    axes.forEach((axis, q) => {
      const idx = Math.max(0, Math.min(n - 1, single && P.index !== undefined ? Math.round(P.index) : n >> 1));
      const s = Math.max(1, Math.min(cellW - 8 * dpr, H - labelH - 4 * dpr));
      const x0 = q * cellW + (cellW - s) / 2, y0 = labelH + (H - labelH - s) / 2;
      ctx.drawImage(this._sliceImage(axis, idx, lut), x0, y0, s, s);
      const [h, v, f] = names[axis];
      ctx.fillStyle = color;
      ctx.fillText(`${h}–${v} slice, ${f} = ${idx}`, q * cellW + cellW / 2, 2 * dpr);
    });
  }
}
