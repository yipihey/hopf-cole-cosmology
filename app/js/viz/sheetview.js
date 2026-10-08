// sheetview.js - Lagrangian phase-space sheet renderer (2D, periodic).
//
// Vertex (i,j) lives at array index i*n + j (i = Lagrangian index along x,
// j = along y) with *unwrapped* Eulerian positions in box units. Each cell
// (i,j),(i+1,j),(i+1,j+1),(i,j+1) is split into the triangles
//   [(i,j),(i+1,j),(i+1,j+1)]  and  [(i,j),(i+1,j+1),(i,j+1)].
// Wrapping at i+1 = n (-> 0) adds +L to x of the wrapped neighbour, at j+1 = n
// adds +L to y, so triangles stay contiguous. Every triangle carries mass
// (L/n)^2 / 2 (mean density 1) and contributes mass/|area| to each pixel it
// covers; overlapping triangles ADD (multi-stream regions).
//
// WebGPU: vertex pulling from a storage buffer of positions; 9 periodic images
// via instancing; additive blending into an rgba16float target; a second pass
// maps accumulated density through the LUT. Fallback: exact CPU rasterization.
//
// P1 option (setVertexWeights + draw({p1:true})): with vertex densities w_v (e.g. 1/|J| at the Lagrangian grid points, index i*n+j,
// same as the positions) a triangle no longer carries the constant density m/|A| but the linear (barycentric) interpolant
//   rho(x) = (m/|A|) (sum_i lambda_i w_i) / mean_i(w_i),
// which still deposits exactly the triangle mass (core/src/sheet.rs::sheet_density_2d_weighted). In the shader the flat per-triangle
// factor m/(|A| mean w) is a flat varying and the vertex weight w a default (linear) varying; the fragment adds factor * w.

import { cmapLUT } from './colormaps.js';
import {
  getGPU, compileModule, validated, surfaceMessage, observeCanvas, pinCssSize,
} from './gpu.js';

const WGSL = /* wgsl */`
struct U {
  view : vec4f,   // x0, y0, size (domain window), L
  geo  : vec4f,   // n, mass per triangle, p1 flag (1 = vertex-interpolated density), 0
  wire : vec4f,   // wire colour rgb, alpha
  rect : vec4f,   // square draw rect in canvas pixels: x0, y0, size, size
  map  : vec4f,   // vmin, vmax (mapping space), log flag, 0
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> pos : array<f32>;
@group(0) @binding(2) var acc : texture_2d<f32>;
@group(0) @binding(3) var lut : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> wts : array<f32>;   // per-vertex density weights (P1), index i*n+j; a dummy when unused

fn vpos(i : u32, j : u32, n : u32) -> vec2f {
  let ii = i % n; let jj = j % n;
  let k = 2u * (ii * n + jj);
  var p = vec2f(pos[k], pos[k + 1u]);
  if (i >= n) { p.x = p.x + u.view.w; }   // periodic wrap in x: add +L
  if (j >= n) { p.y = p.y + u.view.w; }   // periodic wrap in y: add +L
  return p;
}

// position of corner k (0..2) of triangle 'tri'
fn tcorner(tri : u32, k : u32, n : u32) -> vec2f {
  let cell = tri / 2u; let which = tri % 2u;
  let i = cell / n; let j = cell % n;
  var di = 0u; var dj = 0u;
  if (which == 0u) {
    if (k == 1u) { di = 1u; } else if (k == 2u) { di = 1u; dj = 1u; }
  } else {
    if (k == 1u) { di = 1u; dj = 1u; } else if (k == 2u) { dj = 1u; }
  }
  return vpos(i + di, j + dj, n);
}

fn toClip(p : vec2f, inst : u32) -> vec4f {
  let L = u.view.w;
  let off = vec2f(f32(inst % 3u) - 1.0, f32(inst / 3u) - 1.0) * L;
  let w = (p + off - u.view.xy) / u.view.z;        // 0..1 over the view window
  return vec4f(w.x * 2.0 - 1.0, w.y * 2.0 - 1.0, 0.0, 1.0);   // domain y up == texture row 0 at top
}

// weight of corner k (0..2) of triangle 'tri' (P1 sheet)
fn tweight(tri : u32, k : u32, n : u32) -> f32 {
  let cell = tri / 2u; let which = tri % 2u;
  let i = cell / n; let j = cell % n;
  var di = 0u; var dj = 0u;
  if (which == 0u) {
    if (k == 1u) { di = 1u; } else if (k == 2u) { di = 1u; dj = 1u; }
  } else {
    if (k == 1u) { di = 1u; dj = 1u; } else if (k == 2u) { dj = 1u; }
  }
  return wts[((i + di) % n) * n + ((j + dj) % n)];
}

struct TriOut {
  @builtin(position) pos : vec4f,
  @location(0) @interpolate(flat) dens : f32,   // flat per-triangle factor (plain: the density itself)
  @location(1) wi : f32,                        // vertex weight, linearly interpolated (plain: 1)
};

@vertex
fn vsTri(@builtin(vertex_index) vi : u32, @builtin(instance_index) inst : u32) -> TriOut {
  let n = u32(u.geo.x);
  let tri = vi / 3u; let k = vi % 3u;
  let a = tcorner(tri, 0u, n);
  let b = tcorner(tri, 1u, n);
  let c = tcorner(tri, 2u, n);
  let area = 0.5 * abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  var o : TriOut;
  var fac = u.geo.y / max(area, 1e-12);
  var wv = 1.0;
  if (u.geo.z > 0.5) {
    let w0 = tweight(tri, 0u, n); let w1 = tweight(tri, 1u, n); let w2 = tweight(tri, 2u, n);
    let wm = (w0 + w1 + w2) / 3.0;
    if (wm > 0.0) {
      fac = fac / wm;
      wv = w0;
      if (k == 1u) { wv = w1; } else if (k == 2u) { wv = w2; }
    }
  }
  o.dens = min(fac, 6.0e4);   // stay inside half-float range
  o.wi = wv;
  var p = a;
  if (k == 1u) { p = b; } else if (k == 2u) { p = c; }
  o.pos = toClip(p, inst);
  return o;
}

@fragment
fn fsTri(@location(0) @interpolate(flat) dens : f32, @location(1) wi : f32) -> @location(0) vec4f {
  return vec4f(min(dens * wi, 6.0e4), 1.0, 0.0, 0.0);   // r: density sum, g: number of streams
}

// ---- resolve pass: accumulated density -> colour -------------------------
@vertex
fn vsFull(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn lutColor(t : f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 255.0;
  let i0 = i32(floor(x));
  let i1 = min(i0 + 1, 255);
  return mix(textureLoad(lut, vec2<i32>(i0, 0), 0).rgb, textureLoad(lut, vec2<i32>(i1, 0), 0).rgb, x - floor(x));
}

@fragment
fn fsResolve(@builtin(position) fp : vec4f) -> @location(0) vec4f {
  let q = fp.xy - u.rect.xy;
  let S = u.rect.z;
  if (q.x < 0.0 || q.y < 0.0 || q.x >= S || q.y >= S) { return vec4f(0.0); }
  let d = textureLoad(acc, vec2<i32>(q), 0).x;
  var t : f32;
  if (u.map.z > 0.5) {
    if (d <= 0.0) { t = 0.0; } else { t = (log(d) * 0.4342944819 - u.map.x) / (u.map.y - u.map.x); }
  } else {
    t = (d - u.map.x) / (u.map.y - u.map.x);
  }
  return vec4f(lutColor(t), 1.0);
}

// ---- wireframe ------------------------------------------------------------
@vertex
fn vsWire(@builtin(vertex_index) vi : u32, @builtin(instance_index) inst : u32) -> @builtin(position) vec4f {
  let n = u32(u.geo.x);
  let e = vi / 2u; let end = vi % 2u;
  let cell = e / 2u; let dir = e % 2u;
  let i = cell / n; let j = cell % n;
  var di = 0u; var dj = 0u;
  if (dir == 0u) { di = end; } else { dj = end; }
  return toClip(vpos(i + di, j + dj, n), inst);
}

@fragment
fn fsWire() -> @location(0) vec4f {
  return vec4f(u.wire.rgb * u.wire.a, u.wire.a);   // premultiplied
}
`;

const DEFAULTS = {
  mode: 'density', cmap: 'magma', vmin: 0.1, vmax: 100, log: true,
  wireAlpha: 0.35, wireColor: [1, 1, 1], p1: false,
};

export class SheetView {
  /** @param {HTMLCanvasElement} canvas  @param {{cmap?:string}} [opts] */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.backend = null;
    this.positions = null; this.n = 0; this.L = 1;
    this.weights = null;                    // Float32Array n*n of vertex densities (P1) or null
    this.view = null;                       // {x0,y0,size} or null = full box
    this.params = { ...DEFAULTS, ...(opts.cmap ? { cmap: opts.cmap } : {}), ...opts };
    this._initPromise = null;
    this.viewRect = { x: 0, y: 0, size: 0 };
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
      catch (e) { console.error('[viz] SheetView WebGPU init failed, falling back to Canvas2D:', e); this._gpu = null; }
    }
    if (!this.backend) {
      this.ctx2d = canvas.getContext('2d');
      if (!this.ctx2d) { surfaceMessage(canvas, 'SheetView: could not obtain a rendering context.'); throw new Error('no rendering context'); }
      this.backend = 'canvas2d';
      this._off = document.createElement('canvas');
    }
    this._ro = observeCanvas(canvas, (changed) => { if (changed) { this._invalidateTargets(); if (this.positions) this.draw(); } });
  }

  async _initGPU({ device, format }) {
    const module = await compileModule(device, WGSL, 'SheetView');
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const res = await validated(device, async () => {
      // Two layouts: the accumulation/wire passes must not bind the accumulation
      // texture (it is the render attachment there), the resolve pass samples it.
      const layoutA = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: V, buffer: { type: 'read-only-storage' } },
      ] });
      const layoutB = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 2, visibility: F, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: F, texture: { sampleType: 'unfilterable-float' } },
      ] });
      const plA = device.createPipelineLayout({ bindGroupLayouts: [layoutA] });
      const plB = device.createPipelineLayout({ bindGroupLayouts: [layoutB] });
      const accum = device.createRenderPipeline({
        layout: plA,
        vertex: { module, entryPoint: 'vsTri' },
        fragment: { module, entryPoint: 'fsTri', targets: [{ format: 'rgba16float', blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      });
      const resolve = device.createRenderPipeline({
        layout: plB,
        vertex: { module, entryPoint: 'vsFull' },
        fragment: { module, entryPoint: 'fsResolve', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      const wire = device.createRenderPipeline({
        layout: plA,
        vertex: { module, entryPoint: 'vsWire' },
        fragment: { module, entryPoint: 'fsWire', targets: [{ format, blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] },
        primitive: { topology: 'line-list' },
      });
      return { layoutA, layoutB, accum, resolve, wire };
    });
    const ubuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const lutTex = device.createTexture({ size: [256, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('canvas.getContext("webgpu") returned null');
    context.configure({ device, format, alphaMode: 'premultiplied' });
    const wDummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this._gpu = { device, format, ...res, ubuf, lutTex, context, posBuf: null, wBuf: null, wDummy, boundW: null, accTex: null, bindA: null, bindB: null, lutName: null };
    this._ub = new Float32Array(20);
  }

  // -- state ----------------------------------------------------------------

  /**
   * @param {Float32Array|Float64Array} positions interleaved x,y; length 2*n*n; index (i*n+j)
   * @param {number} n  grid size
   * @param {number} L  box size
   */
  setMesh(positions, n, L) {
    if (positions.length < 2 * n * n) throw new Error(`SheetView.setMesh: need ${2 * n * n} floats, got ${positions.length}`);
    if (!(positions instanceof Float32Array)) positions = Float32Array.from(positions);
    const sizeChanged = n !== this.n;
    this.positions = positions; this.n = n; this.L = L;
    const G = this._gpu;
    if (G) {
      const bytes = 8 * n * n;
      if (bytes > G.device.limits.maxStorageBufferBindingSize) { surfaceMessage(this.canvas, `SheetView: mesh of ${n}x${n} exceeds GPU buffer limit.`); throw new Error('mesh too large'); }
      if (!G.posBuf || sizeChanged) {
        if (G.posBuf) G.posBuf.destroy();
        G.posBuf = G.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        G.bindA = G.bindB = null;
      }
      G.device.queue.writeBuffer(G.posBuf, 0, positions.buffer, positions.byteOffset, bytes);
    }
    if (this.weights && this.weights.length !== n * n) this.weights = null;      // grid size changed under the weights
    if (G && this.weights !== null && G.wBuf && G.wBuf.size !== 4 * n * n) this._uploadWeights();
  }

  /**
   * Per-vertex density weights for the P1 sheet (draw option p1: true): Float32Array of length n*n, index i*n+j like the positions
   * (e.g. 1/|J| on the Lagrangian grid); null removes them. Without weights, p1 has no effect.
   */
  setVertexWeights(w) {
    if (w == null) { this.weights = null; return; }
    if (!this.n || w.length < this.n * this.n) throw new Error(`SheetView.setVertexWeights: need ${this.n * this.n} floats (call setMesh first), got ${w.length}`);
    this.weights = w instanceof Float32Array ? w.subarray(0, this.n * this.n) : Float32Array.from(w.subarray ? w.subarray(0, this.n * this.n) : w.slice(0, this.n * this.n));
    this._uploadWeights();
  }

  _uploadWeights() {
    const G = this._gpu;
    if (!G || !this.weights) return;
    const bytes = 4 * this.n * this.n;
    if (!G.wBuf || G.wBuf.size !== bytes) {
      if (G.wBuf) G.wBuf.destroy();
      G.wBuf = G.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      G.bindA = null;
    }
    G.device.queue.writeBuffer(G.wBuf, 0, this.weights.buffer, this.weights.byteOffset, bytes);
  }

  /** Zoom to a square window of the periodic domain; call with no args to reset. */
  setView(x0, y0, size) { this.view = x0 === undefined ? null : { x0, y0, size }; }

  _invalidateTargets() {
    const G = this._gpu;
    if (G && G.accTex) { G.accTex.destroy(); G.accTex = null; G.bindA = G.bindB = null; }
  }

  destroy() {
    if (this._ro) this._ro.disconnect();
    const G = this._gpu;
    if (G) { if (G.posBuf) G.posBuf.destroy(); if (G.wBuf) G.wBuf.destroy(); if (G.wDummy) G.wDummy.destroy(); if (G.accTex) G.accTex.destroy(); }
  }

  /**
   * Render. Parameters persist between calls (and are reused on resize).
   * @param {{mode?:'density'|'wire'|'both', cmap?:string, vmin?:number, vmax?:number, log?:boolean, wireAlpha?:number, wireColor?:number[], p1?:boolean}} [p]
   *   p1: vertex-interpolated (P1) density; needs setVertexWeights, otherwise the plain constant-per-triangle density is drawn
   */
  draw(p = {}) {
    if (!this.backend || !this.positions) return;
    Object.assign(this.params, p);
    try {
      const W = this.canvas.width, H = this.canvas.height, S = Math.max(1, Math.min(W, H));
      this.viewRect = { x: Math.floor((W - S) / 2), y: Math.floor((H - S) / 2), size: S };
      if (this.backend === 'webgpu') this._drawGPU(); else this._draw2D();
    } catch (e) {
      surfaceMessage(this.canvas, 'SheetView draw failed: ' + e.message);
    }
  }

  _window() {
    const v = this.view;
    return v ? [v.x0, v.y0, v.size] : [0, 0, this.L];
  }

  // -- WebGPU ---------------------------------------------------------------

  _drawGPU() {
    const G = this._gpu, { device } = G, P = this.params, R = this.viewRect, n = this.n, L = this.L;
    const wantDensity = P.mode !== 'wire', wantWire = P.mode !== 'density';
    if (!G.accTex || G.accTex.width !== R.size) {
      if (G.accTex) G.accTex.destroy();
      G.accTex = device.createTexture({ size: [R.size, R.size], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      G.bindA = G.bindB = null;
    }
    if (G.lutName !== P.cmap) {
      device.queue.writeTexture({ texture: G.lutTex }, cmapLUT(P.cmap, 256), { bytesPerRow: 1024 }, { width: 256, height: 1 });
      G.lutName = P.cmap;
    }
    const useP1 = !!(P.p1 && this.weights);
    const wRes = useP1 && G.wBuf ? G.wBuf : G.wDummy;
    if (G.boundW !== wRes) { G.bindA = null; G.boundW = wRes; }
    if (!G.bindA) {
      G.bindA = device.createBindGroup({ layout: G.layoutA, entries: [
        { binding: 0, resource: { buffer: G.ubuf } },
        { binding: 1, resource: { buffer: G.posBuf } },
        { binding: 4, resource: { buffer: wRes } },
      ] });
      G.bindB = device.createBindGroup({ layout: G.layoutB, entries: [
        { binding: 0, resource: { buffer: G.ubuf } },
        { binding: 2, resource: G.accTex.createView() },
        { binding: 3, resource: G.lutTex.createView() },
      ] });
    }
    const [x0, y0, size] = this._window();
    const lo = P.log ? Math.log10(Math.max(P.vmin, 1e-30)) : P.vmin, hi = P.log ? Math.log10(Math.max(P.vmax, 1e-30)) : P.vmax;
    const u = this._ub, wc = P.wireColor || [1, 1, 1];
    u.set([x0, y0, size, L,  n, (L / n) * (L / n) / 2, useP1 ? 1 : 0, 0,  wc[0], wc[1], wc[2], P.wireAlpha,
      R.x, R.y, R.size, R.size,  lo, hi, P.log ? 1 : 0, 0]);
    device.queue.writeBuffer(G.ubuf, 0, u);

    const enc = device.createCommandEncoder();
    if (wantDensity) {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: G.accTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(G.accum);
      pass.setBindGroup(0, G.bindA);
      pass.draw(6 * n * n, 9);       // 2 triangles per cell x 3 vertices, 9 periodic images
      pass.end();
    }
    const bg = wantDensity ? { r: 0, g: 0, b: 0, a: 0 } : { r: 0.04, g: 0.05, b: 0.07, a: 1 };
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: G.context.getCurrentTexture().createView(), clearValue: bg, loadOp: 'clear', storeOp: 'store' }] });
    if (wantDensity) { pass.setPipeline(G.resolve); pass.setBindGroup(0, G.bindB); pass.draw(3); }
    if (wantWire) {
      pass.setBindGroup(0, G.bindA);
      pass.setViewport(R.x, R.y, R.size, R.size, 0, 1);
      pass.setScissorRect(R.x, R.y, R.size, R.size);
      pass.setPipeline(G.wire);
      pass.draw(4 * n * n, 9);       // 2 edges per cell x 2 vertices
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // -- Canvas2D fallback (exact CPU rasterization) ------------------------------

  /** Corner positions of cell (i,j): [a,b,c,d] x/y arrays with periodic +L fix-ups. */
  _cell(i, j, out) {
    const { positions: P, n, L } = this;
    const i1 = i + 1 === n ? 0 : i + 1, j1 = j + 1 === n ? 0 : j + 1;
    const ox = i + 1 === n ? L : 0, oy = j + 1 === n ? L : 0;
    const ka = 2 * (i * n + j), kb = 2 * (i1 * n + j), kc = 2 * (i1 * n + j1), kd = 2 * (i * n + j1);
    out[0] = P[ka];      out[1] = P[ka + 1];
    out[2] = P[kb] + ox; out[3] = P[kb + 1];
    out[4] = P[kc] + ox; out[5] = P[kc + 1] + oy;
    out[6] = P[kd];      out[7] = P[kd + 1] + oy;
  }

  _draw2D() {
    const ctx = this.ctx2d, P = this.params, R = this.viewRect, n = this.n, L = this.L;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const wantDensity = P.mode !== 'wire', wantWire = P.mode !== 'density';
    const [wx0, wy0, wsize] = this._window();
    const N = Math.min(R.size, 1400);              // raster resolution
    const scale = N / wsize;
    if (!wantDensity) { ctx.fillStyle = '#0a0d12'; ctx.fillRect(0, 0, W, H); }
    const shifts = [];
    for (let sy = -1; sy <= 1; sy++) for (let sx = -1; sx <= 1; sx++) shifts.push([sx * L, sy * L]);
    const cell = new Float64Array(8);
    const useP1 = !!(P.p1 && this.weights), Wt = this.weights;

    if (wantDensity) {
      const acc = new Float32Array(N * N);
      const mass = (L / n) * (L / n) / 2;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        this._cell(i, j, cell);
        // vertex weights of the cell corners a,b,c,d (P1)
        const i1 = i + 1 === n ? 0 : i + 1, j1 = j + 1 === n ? 0 : j + 1;
        const wa = useP1 ? Wt[i * n + j] : 1, wb = useP1 ? Wt[i1 * n + j] : 1, wc = useP1 ? Wt[i1 * n + j1] : 1, wd = useP1 ? Wt[i * n + j1] : 1;
        for (let t = 0; t < 2; t++) {
          // triangle 0: a,b,c ; triangle 1: a,c,d
          const ax = cell[0], ay = cell[1];
          const bx = t ? cell[4] : cell[2], by = t ? cell[5] : cell[3];
          const cx = t ? cell[6] : cell[4], cy = t ? cell[7] : cell[5];
          const det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
          const area = 0.5 * Math.abs(det);
          if (!(area > 1e-12)) continue;
          // P1: flat factor m / (|A| mean w) times the linearly interpolated vertex weight; plain: the constant m / |A|
          const wB = t ? wc : wb, wC = t ? wd : wc;
          const wmean = (wa + wB + wC) / 3;
          const p1t = useP1 && wmean > 0;
          const dens = p1t ? mass / area / wmean : Math.min(mass / area, 6e4);
          const minx = Math.min(ax, bx, cx), maxx = Math.max(ax, bx, cx), miny = Math.min(ay, by, cy), maxy = Math.max(ay, by, cy);
          for (const [sx, sy] of shifts) {
            if (maxx + sx < wx0 || minx + sx > wx0 + wsize || maxy + sy < wy0 || miny + sy > wy0 + wsize) continue;
            // pixel coordinates: x right, y down (domain y up)
            const pax = (ax + sx - wx0) * scale, pay = N - (ay + sy - wy0) * scale;
            const pbx = (bx + sx - wx0) * scale, pby = N - (by + sy - wy0) * scale;
            const pcx = (cx + sx - wx0) * scale, pcy = N - (cy + sy - wy0) * scale;
            const idet = 1 / ((pbx - pax) * (pcy - pay) - (pby - pay) * (pcx - pax));
            const xlo = Math.max(0, Math.floor(Math.min(pax, pbx, pcx) - 0.5)), xhi = Math.min(N - 1, Math.ceil(Math.max(pax, pbx, pcx)));
            const ylo = Math.max(0, Math.floor(Math.min(pay, pby, pcy) - 0.5)), yhi = Math.min(N - 1, Math.ceil(Math.max(pay, pby, pcy)));
            for (let py = ylo; py <= yhi; py++) {
              const qy = py + 0.50027 - pay;
              for (let px = xlo; px <= xhi; px++) {
                const qx = px + 0.50013 - pax;
                // barycentric coordinates relative to a
                const l1 = (qx * (pcy - pay) - qy * (pcx - pax)) * idet;
                const l2 = ((pbx - pax) * qy - (pby - pay) * qx) * idet;
                if (l1 >= 0 && l2 >= 0 && l1 + l2 <= 1) acc[py * N + px] += p1t ? Math.min(dens * ((1 - l1 - l2) * wa + l1 * wB + l2 * wC), 6e4) : dens;
              }
            }
          }
        }
      }
      // colour map
      const off = this._off; off.width = N; off.height = N;
      const octx = off.getContext('2d'), img = octx.createImageData(N, N), px = img.data, lut = cmapLUT(P.cmap, 256);
      const lo = P.log ? Math.log10(Math.max(P.vmin, 1e-30)) : P.vmin, hi = P.log ? Math.log10(Math.max(P.vmax, 1e-30)) : P.vmax;
      for (let k = 0; k < N * N; k++) {
        const d = acc[k];
        let tt = P.log ? (d > 0 ? (Math.log10(d) - lo) / (hi - lo) : 0) : (d - lo) / (hi - lo);
        tt = tt > 0 ? (tt < 1 ? tt : 1) : 0;
        const q = 4 * Math.round(tt * 255);
        px[4 * k] = lut[q]; px[4 * k + 1] = lut[q + 1]; px[4 * k + 2] = lut[q + 2]; px[4 * k + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, R.x, R.y, R.size, R.size);
    }

    if (wantWire) {
      const k = R.size / wsize;
      ctx.save();
      ctx.beginPath(); ctx.rect(R.x, R.y, R.size, R.size); ctx.clip();
      const wc = P.wireColor || [1, 1, 1];
      ctx.strokeStyle = `rgba(${wc[0] * 255 | 0},${wc[1] * 255 | 0},${wc[2] * 255 | 0},${P.wireAlpha})`;
      ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1) * 0.75;
      ctx.beginPath();
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        this._cell(i, j, cell);
        // edges a->b and a->d
        for (const [sx, sy] of shifts) {
          const ax = cell[0] + sx, ay = cell[1] + sy;
          const bx = cell[2] + sx, by = cell[3] + sy, dx = cell[6] + sx, dy = cell[7] + sy;
          if (Math.max(ax, bx, dx) < wx0 || Math.min(ax, bx, dx) > wx0 + wsize || Math.max(ay, by, dy) < wy0 || Math.min(ay, by, dy) > wy0 + wsize) continue;
          const X = (x) => R.x + (x - wx0) * k, Y = (y) => R.y + R.size - (y - wy0) * k;
          ctx.moveTo(X(bx), Y(by)); ctx.lineTo(X(ax), Y(ay)); ctx.lineTo(X(dx), Y(dy));
        }
      }
      ctx.stroke();
      ctx.restore();
    }
  }
}
