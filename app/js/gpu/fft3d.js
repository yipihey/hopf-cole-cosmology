// fft3d.js - 3-D complex FFT on the GPU (Stockham radix-2 auto-sort, one dispatch per stage).
//
// Data: interleaved complex f32 in a storage buffer of 2*n^3 floats (vec2<f32> per cell),
// index (ix*n + iy)*n + iz, the same layout as the CPU fields. n is a power of two (32..256).
//
//   const fft = new GpuFFT3D(device, n); await fft.init();
//   fft.forward(src, dst [, encoder]);   // dst = FFT(src), unnormalised, exp(-i k.x)
//   fft.inverse(src, dst [, encoder]);   // dst = IFFT(src) / n^3
//   fft.packReal(realBuf, cplxBuf [, encoder, offset]);   // cplx = (real - offset) + 0i
//   fft.unpackReal(cplxBuf, realBuf [, encoder]);          // real = Re(cplx)
//
// If `encoder` is omitted each call records, finishes and submits its own command buffer;
// otherwise the passes are appended to the given GPUCommandEncoder (the caller submits).
// src and dst of forward/inverse must be distinct buffers; src is never modified.
// A call records 3*log2(n) compute passes (x axis first, then y, then z) ping-ponging between
// src, an internal scratch buffer and dst. The 1/n^3 inverse normalisation is folded into the
// last stage.  Twiddles come from a table computed in double precision on the CPU.

import { ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR } from './util.js';

const FFT_WGSL = /* wgsl */`
struct P { n: u32, stride: u32, ns: u32, twstep: u32, sign: f32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;
${WGSL_LINEAR}
@compute @workgroup_size(256)
fn stage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = linear_id(gid, nwg, 256u);
  let n = p.n;
  let half = n >> 1u;
  let total = (n * n * n) >> 1u;
  if (t >= total) { return; }
  let nl = n * n;
  var l: u32; var j: u32;
  if (p.stride == 1u) { l = t / half; j = t % half; } else { l = t % nl; j = t / nl; }
  let lo = l % p.stride;
  let hi = l / p.stride;
  let base = hi * p.stride * n + lo;
  let k = j & (p.ns - 1u);
  let v0 = src[base + j * p.stride];
  let v1r = src[base + (j + half) * p.stride];
  let w = tw[k * p.twstep];
  let c = vec2<f32>(w.x, p.sign * w.y);
  let v1 = vec2<f32>(v1r.x * c.x - v1r.y * c.y, v1r.x * c.y + v1r.y * c.x);
  let jo = 2u * j - k;
  dst[base + jo * p.stride] = (v0 + v1) * p.scale;
  dst[base + (jo + p.ns) * p.stride] = (v0 - v1) * p.scale;
}
`;

const PACK_WGSL = /* wgsl */`
struct Q { count: u32, off: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> q: Q;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<vec2<f32>>;
${WGSL_LINEAR}
@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= q.count) { return; }
  b[i] = vec2<f32>(a[i] - q.off, 0.0);
}
`;
const UNPACK_WGSL = /* wgsl */`
struct Q { count: u32, off: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> q: Q;
@group(0) @binding(1) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
${WGSL_LINEAR}
@compute @workgroup_size(256)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= q.count) { return; }
  b[i] = a[i].x;
}
`;

const pipeCache = new WeakMap();
function getPipelines(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const [m1, m2, m3] = await Promise.all([compile(device, FFT_WGSL, 'fft stage'), compile(device, PACK_WGSL, 'fft pack'), compile(device, UNPACK_WGSL, 'fft unpack')]);
      return {
        stage: makePipeline(device, m1, 'stage', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RO_ENTRY(3)], 'fft stage'),
        pack: makePipeline(device, m2, 'pack', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'fft pack'),
        unpack: makePipeline(device, m3, 'unpack', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'fft unpack'),
      };
    })();
    pipeCache.set(device, p);
  }
  return p;
}

const SLOT = 256;

export class GpuFFT3D {
  constructor(device, n) {
    if (!Number.isInteger(Math.log2(n)) || n < 4) throw new Error('GpuFFT3D: n must be a power of two');
    this.device = device; this.n = n; this.size = n * n * n;
    this.L = Math.log2(n);
    this.bytes = 2 * 4 * this.size;
    const lim = device.limits;
    if (this.bytes > lim.maxStorageBufferBindingSize || this.bytes > lim.maxBufferSize) {
      throw new Error(`GpuFFT3D: ${n}^3 complex buffer (${(this.bytes / 1048576).toFixed(0)} MiB) exceeds the adapter limit maxStorageBufferBindingSize = ${(lim.maxStorageBufferBindingSize / 1048576).toFixed(0)} MiB`);
    }
    this.ring = ParamRing.get(device);
    this.scratch = null;
  }

  async init() {
    const d = this.device, n = this.n, L = this.L;
    this.pipes = await getPipelines(d);
    // twiddle table: exp(+2 pi i m / n), m = 0 .. n/2-1 (double precision on the CPU)
    const tw = new Float32Array(n);
    for (let m = 0; m < n / 2; m++) { tw[2 * m] = Math.cos(2 * Math.PI * m / n); tw[2 * m + 1] = Math.sin(2 * Math.PI * m / n); }
    this.tw = d.createBuffer({ label: 'fft twiddles', size: Math.max(16, tw.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.tw, 0, tw);
    // stage parameters: slot = ((dir*3 + axisIdx) * L + stage)
    const nslots = 2 * 3 * L;
    const raw = new ArrayBuffer(nslots * SLOT);
    const u = new Uint32Array(raw), f = new Float32Array(raw);
    const strides = [n * n, n, 1];
    for (let dir = 0; dir < 2; dir++) for (let ax = 0; ax < 3; ax++) for (let s = 0; s < L; s++) {
      const o = (((dir * 3 + ax) * L) + s) * (SLOT / 4);
      const ns = 1 << s;
      u[o] = n; u[o + 1] = strides[ax]; u[o + 2] = ns; u[o + 3] = n / (2 * ns);
      f[o + 4] = dir === 0 ? -1 : 1;
      const last = ax === 2 && s === L - 1;
      f[o + 5] = (dir === 1 && last) ? 1 / (n * n * n) : 1;
    }
    this.params = d.createBuffer({ label: 'fft params', size: raw.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.params, 0, raw);
    this.scratch = d.createBuffer({ label: 'fft scratch', size: this.bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.bgCache = new WeakMap();
    return this;
  }

  _bg(a, b) {
    // bind groups keyed on the buffer pair; WeakMap keyed on the destination buffer
    let m = this.bgCache.get(a);
    if (!m) { m = new WeakMap(); this.bgCache.set(a, m); }
    let g = m.get(b);
    if (!g) {
      g = makeBindGroup(this.device, this.pipes.stage.bgl, [{ buffer: this.params, size: 32 }, a, b, this.tw], 'fft bg');
      m.set(b, g);
    }
    return g;
  }

  _run(src, dst, dir, encoder) {
    if (src === dst) throw new Error('GpuFFT3D: src and dst must be different buffers');
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'fft' });
    const L = this.L, T = 3 * L;
    const total = this.size / 2;
    // stage i writes into dst when (T-1-i) is even, else into scratch (so the last stage lands in dst)
    let cur = src;
    const pass = enc.beginComputePass({ label: dir ? 'ifft3d' : 'fft3d' });
    pass.setPipeline(this.pipes.stage.pipeline);
    for (let i = 0; i < T; i++) {
      const out = ((T - 1 - i) % 2 === 0) ? dst : this.scratch;
      const ax = Math.floor(i / L), s = i % L;
      pass.setBindGroup(0, this._bg(cur, out), [(((dir * 3 + ax) * L) + s) * SLOT]);
      dispatch1d(pass, total, 256);
      cur = out;
    }
    pass.end();
    if (own) this.device.queue.submit([enc.finish()]);
  }

  forward(src, dst, encoder = null) { this._run(src, dst, 0, encoder); }
  inverse(src, dst, encoder = null) { this._run(src, dst, 1, encoder); }

  _simple(kind, a, b, encoder, off) {
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder();
    const P = this.pipes[kind];
    const slot = this.ring.write([['u', this.size], off]);
    const pass = enc.beginComputePass({ label: kind });
    pass.setPipeline(P.pipeline);
    pass.setBindGroup(0, makeBindGroup(this.device, P.bgl, [this.ring.resource(16), a, b], kind), [slot]);
    dispatch1d(pass, this.size, 256);
    pass.end();
    if (own) this.device.queue.submit([enc.finish()]);
  }
  /** cplx[i] = (real[i] - offset) + 0i */
  packReal(realBuf, cplxBuf, encoder = null, offset = 0) { this._simple('pack', realBuf, cplxBuf, encoder, offset); }
  /** real[i] = Re(cplx[i]) */
  unpackReal(cplxBuf, realBuf, encoder = null) { this._simple('unpack', cplxBuf, realBuf, encoder, 0); }

  destroy() {
    for (const b of [this.tw, this.params, this.scratch]) if (b) b.destroy();
    this.tw = this.params = this.scratch = null;
    this.bgCache = new WeakMap();
  }
}
