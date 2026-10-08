// exact2d.js - exact (conservative, clipping-based) sheet deposition on the GPU in 2-D: every triangle of the displaced Lagrangian
// grid is clipped against the Eulerian cells it overlaps and deposits the exact integral of its density profile.
//
//   const ex = new GpuExact2D(device); await ex.init();
//   rho = ex.density(positions, weights|null, ne)   GPUBuffer ne*ne f32 (persistent, [ix*ne+iy]), cell-averaged rho/rhobar
//   rho = await exactDensity2D(device, positions, weights, ne)       one-shot helper (positions/weights may be Float32Arrays)
//
// positions: interleaved (x, y) in box units of the n^2 Lagrangian vertices [i*n+j] (x = q + Psi, unwrapped, like
// CosmoSim.positions); weights: n^2 vertex densities 1/|J| (P1 shape) or null (P0, constant density per triangle).
// The grid is periodic: the vertices with i + 1 = n (or j + 1 = n) are the wrapped vertices plus one box length, exactly as in
// core/src/sheet.rs::sheet_density_2d_exact, and the deposit wraps the cell index modulo ne.
//
// One thread per Lagrangian cell (two triangles).  A triangle is clipped, for each cell of its bounding box, by Sutherland-Hodgman
// against the cell's x-slab and then y-slab (at most 7 vertices); area and first moments come from the exact Green's-theorem edge sums
// (core/src/r3d.rs::Poly<2>::moments, order <= 1) about the lower corner of the cell.
//   P0:  deposit m = (ne/n)^2 / (2 A) * m0           (A = triangle area in cell units; the triangle has mass (ne/n)^2/2 in units
//                                                       of the mean cell mass)
//   P1:  deposit m = (ne/n)^2 / (2 A) * m0 * sum_c w_c lambda_c(centroid of the piece) / mean(w)   (lambda affine => exact)
// as 18-bit fixed-point u32 atomics (CIC_SCALE = 2^18 per unit rho/rhobar, a cell overflows at rho/rhobar = 16384, like the CIC and
// sheet kernels).  Triangles whose bounding box spans more than MAXB cells per axis are skipped and counted (stats()).
//
// Positions and the base cell are handled in cell units RELATIVE TO THE FIRST VERTEX'S CELL (integer base added back), so the f32
// kernel works on O(1) numbers.

import { ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, readU32, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';

export const EXACT2D_SCALE = 1 << 18;
export const EXACT2D_MAXB = 24;

export const WGSL_CLIP2D = /* wgsl */`
const CAP2: u32 = 10u;
struct Poly2 { p: array<vec2<f32>, CAP2>, f: array<f32, CAP2>, n: u32 };   // f: scalar attribute per vertex (linear inside)
var<private> Q_T: Poly2;
var<private> Q_X: Poly2;
var<private> Q_A: Poly2;
var<private> Q_B: Poly2;

// Sutherland-Hodgman half-plane clip: keep sgn * (x[ax] - coord) <= 0
fn clip2(src: ptr<private, Poly2>, dst: ptr<private, Poly2>, ax: u32, coord: f32, sgn: f32) -> u32 {
  let n = (*src).n;
  var m = 0u;
  for (var i = 0u; i < n; i++) {
    let j = select(i + 1u, 0u, i + 1u == n);
    let a = (*src).p[i];
    let b = (*src).p[j];
    let sa = sgn * (a[ax] - coord);
    let sb = sgn * (b[ax] - coord);
    if (sa <= 0.0) { (*dst).p[m] = a; (*dst).f[m] = (*src).f[i]; m++; }
    if ((sa <= 0.0) != (sb <= 0.0)) {
      let t = sa / (sa - sb);
      (*dst).p[m] = a + (b - a) * t;
      (*dst).f[m] = (*src).f[i] + ((*src).f[j] - (*src).f[i]) * t;
      m++;
    }
  }
  (*dst).n = m;
  return select(0u, m, m >= 3u);
}
fn slab2(src: ptr<private, Poly2>, dst: ptr<private, Poly2>, ax: u32, lo: f32, hi: f32) -> u32 {
  if (clip2(src, &Q_A, ax, lo, -1.0) == 0u) { return 0u; }
  return clip2(&Q_A, dst, ax, hi, 1.0);
}
// (area, first moments) about 'org' (fixtures only) from the edge sums of Green's theorem
fn moments2(p: ptr<private, Poly2>, org: vec2<f32>) -> vec3<f32> {
  let n = (*p).n;
  var a = 0.0;
  var m = vec2<f32>(0.0);
  for (var i = 0u; i < n; i++) {
    let p0 = (*p).p[i] - org;
    let p1 = (*p).p[select(i + 1u, 0u, i + 1u == n)] - org;
    let c = p0.x * p1.y - p1.x * p0.y;
    a += c;
    m += c * (p0 + p1);
  }
  return vec3<f32>(a / 2.0, m / 6.0);
}
// (area, integral of the linear scalar f) of the polygon, fanned from its vertex 0 (f is interpolated, never extrapolated)
fn integ2(p: ptr<private, Poly2>) -> vec2<f32> {
  let n = (*p).n;
  let o = (*p).p[0];
  let f0 = (*p).f[0];
  var a = 0.0;
  var ia = 0.0;
  for (var i = 1u; i + 1u < n; i++) {
    let u = (*p).p[i] - o;
    let v = (*p).p[i + 1u] - o;
    let ar = u.x * v.y - u.y * v.x;
    a += ar;
    ia += ar * (f0 + (*p).f[i] + (*p).f[i + 1u]);
  }
  return vec2<f32>(a / 2.0, ia / 6.0);
}
// cells [blo, bhi) (relative to the base cell) overlapped by the triangle Q_T
fn walk_tri(blo: vec2<i32>, bhi: vec2<i32>) {
  for (var ci = blo.x; ci < bhi.x; ci++) {
    if (slab2(&Q_T, &Q_X, 0u, f32(ci), f32(ci + 1)) == 0u) { continue; }
    var ylo = Q_X.p[0].y; var yhi = ylo;
    for (var v = 1u; v < Q_X.n; v++) { ylo = min(ylo, Q_X.p[v].y); yhi = max(yhi, Q_X.p[v].y); }
    let j0 = max(blo.y, i32(floor(ylo)));
    let j1 = min(bhi.y, i32(ceil(yhi)));
    for (var cj = j0; cj < j1; cj++) {
      if (slab2(&Q_X, &Q_B, 1u, f32(cj), f32(cj + 1)) == 0u) { continue; }
      emit(vec2<i32>(ci, cj), &Q_B);
    }
  }
}
fn init_tri(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, f: vec3<f32>) {   // counter-clockwise
  Q_T.p[0] = a; Q_T.p[1] = b; Q_T.p[2] = c; Q_T.f[0] = f.x; Q_T.f[1] = f.y; Q_T.f[2] = f.z; Q_T.n = 3u;
}
`;

const WGSL_MAIN = /* wgsl */`
${WGSL_LINEAR}
${WGSL_CLIP2D}
struct XU { count: u32, n: u32, ne: u32, scale: u32, useW: u32, maxb: u32, base: u32, pad: u32, nef: f32, rat: f32 };
@group(0) @binding(0) var<uniform> xu: XU;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> wv: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

var<private> c_base: vec2<i32>;
var<private> c_dens: f32;

fn wrapi(c: i32, ne: i32) -> u32 { return u32(((c % ne) + ne) % ne); }

fn emit(cell: vec2<i32>, p: ptr<private, Poly2>) {
  // (area, int f) of the piece, f = (sum_c w_c lambda_c) / mean(w) carried (and interpolated) by the polygon vertices
  let r = integ2(p);
  if (r.x <= 1e-12) { return; }
  let q = u32(min(c_dens * max(r.y, 0.0) * f32(xu.scale) + 0.5, 2147483648.0));
  let ne = i32(xu.ne);
  let c = c_base + cell;
  atomicAdd(&acc[wrapi(c.x, ne) * xu.ne + wrapi(c.y, ne)], q);
}

@compute @workgroup_size(64)
fn exact_tri(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 64u) + xu.base;
  if (i >= xu.count) { return; }
  let n = xu.n;
  let ix = i / n; let iy = i % n;
  // corners 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1) in absolute cell units
  var cp: array<vec2<f32>, 4>;
  var cw: array<f32, 4>;
  for (var b = 0u; b < 4u; b++) {
    let bx = b & 1u; let by = b >> 1u;
    let jx = (ix + bx) % n; let jy = (iy + by) % n;
    let idx = jx * n + jy;
    var p = vec2<f32>(pos[2u * idx], pos[2u * idx + 1u]) * xu.nef;
    if (ix + bx == n) { p.x += xu.nef; }
    if (iy + by == n) { p.y += xu.nef; }
    cp[b] = p;
    cw[b] = select(1.0, wv[idx], xu.useW != 0u);
  }
  let bf = floor(cp[0]);
  c_base = vec2<i32>(bf);
  for (var b = 0u; b < 4u; b++) { cp[b] = cp[b] - bf; }
  for (var t = 0u; t < 2u; t++) {
    var a = cp[0]; var b = cp[1]; var c = cp[3];
    var wa = cw[0]; var wb = cw[1]; var wc = cw[3];
    if (t == 1u) { b = cp[3]; c = cp[2]; wb = cw[3]; wc = cw[2]; }
    var e1 = b - a; var e2 = c - a;
    var area2 = e1.x * e2.y - e1.y * e2.x;
    if (abs(area2) < 1e-12) { atomicAdd(&stats[2], 1u); continue; }
    if (area2 < 0.0) {
      let tp = b; b = c; c = tp; let tw = wb; wb = wc; wc = tw;
      e1 = b - a; e2 = c - a; area2 = -area2;
    }
    var phi = vec3<f32>(1.0);
    if (xu.useW != 0u) {
      let wm = (wa + wb + wc) / 3.0;
      if (wm > 0.0) { phi = vec3<f32>(wa, wb, wc) / wm; }
    }
    c_dens = xu.rat * xu.rat / area2;
    let lo = min(a, min(b, c)); let hi = max(a, max(b, c));
    let blo = vec2<i32>(floor(lo));
    let bhi = max(vec2<i32>(ceil(hi)), blo + vec2<i32>(1));
    let ext = bhi - blo;
    if (max(ext.x, ext.y) > i32(xu.maxb)) { atomicAdd(&stats[0], 1u); continue; }
    init_tri(a, b, c, phi);
    walk_tri(blo, bhi);
  }
}
`;

const WGSL_CONV = /* wgsl */`
${WGSL_LINEAR}
struct CU { count: u32, scale: u32 };
@group(0) @binding(0) var<uniform> cu: CU;
@group(0) @binding(1) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> rho: array<f32>;
@compute @workgroup_size(64)
fn exact_conv(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 64u);
  if (i >= cu.count) { return; }
  rho[i] = f32(atomicLoad(&acc[i])) / f32(cu.scale);
}
`;

const WGSL_FIX = /* wgsl */`
${WGSL_LINEAR}
${WGSL_CLIP2D}
struct FU { ncases: u32, maxb: u32, maxrec: u32 };
@group(0) @binding(0) var<uniform> fu: FU;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read_write> fcnt: array<atomic<u32>>;
var<private> c_case: u32;
fn emit(cell: vec2<i32>, p: ptr<private, Poly2>) {
  let m = moments2(p, vec2<f32>(cell));
  let r = atomicAdd(&fcnt[0], 1u);
  if (r >= fu.maxrec) { return; }
  let o = 8u * r;
  fout[o] = f32(c_case); fout[o + 1u] = f32(cell.x); fout[o + 2u] = f32(cell.y);
  fout[o + 4u] = m.x; fout[o + 5u] = m.y; fout[o + 6u] = m.z;
}
@compute @workgroup_size(32)
fn fix_tri(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 32u);
  if (i >= fu.ncases) { return; }
  c_case = i;
  var v: array<vec2<f32>, 3>;
  for (var k = 0u; k < 3u; k++) { v[k] = vec2<f32>(fin[6u * i + 2u * k], fin[6u * i + 2u * k + 1u]); }
  let det = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
  if (det < 0.0) { let t = v[1]; v[1] = v[2]; v[2] = t; }
  init_tri(v[0], v[1], v[2], vec3<f32>(1.0));
  let lo = min(v[0], min(v[1], v[2])); let hi = max(v[0], max(v[1], v[2]));
  let blo = vec2<i32>(floor(lo));
  let bhi = max(vec2<i32>(ceil(hi)), blo + vec2<i32>(1));
  let ext = bhi - blo;
  if (max(ext.x, ext.y) > i32(fu.maxb)) { atomicAdd(&fcnt[1], 1u); return; }
  walk_tri(blo, bhi);
}
`;

const pipeCache = new WeakMap();
function getPipelines(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const [mm, mc] = await Promise.all([compile(device, WGSL_MAIN, 'exact2d'), compile(device, WGSL_CONV, 'exact2d convert')]);
      return {
        tri: makePipeline(device, mm, 'exact_tri', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RO_ENTRY(3), RW_ENTRY(4)], 'exact_tri'),
        conv: makePipeline(device, mc, 'exact_conv', [U_ENTRY(0), RW_ENTRY(1), RW_ENTRY(2)], 'exact_conv'),
      };
    })();
    pipeCache.set(device, p);
  }
  return p;
}

export class GpuExact2D {
  constructor(device) { this.device = device; this.ring = ParamRing.get(device); this.bufs = new Map(); }
  async init() {
    const p = await getPipelines(this.device);
    this.pipe = p.tri; this.conv = p.conv;
    this.dummy = this.device.createBuffer({ label: 'exact2d no weights', size: 16, usage: STORAGE_RW });
    return this;
  }
  destroy() {
    for (const b of this.bufs.values()) b.destroy();
    this.bufs.clear();
    if (this.dummy) { this.dummy.destroy(); this.dummy = null; }
  }
  _buf(name, bytes) {
    let b = this.bufs.get(name);
    if (!b || b.size < bytes) {
      if (b) b.destroy();
      b = this.device.createBuffer({ label: 'exact2d ' + name, size: Math.max(16, bytes), usage: STORAGE_RW });
      this.bufs.set(name, b);
    }
    return b;
  }

  /**
   * Cell-averaged rho/rhobar on ne x ne from the exact triangle deposit (persistent buffer 'rho' of this object).
   * `positions`: GPUBuffer of interleaved (x, y) f32 (n^2 vertices, n = sqrt(size / 8)); `weights`: GPUBuffer of n^2 vertex densities
   * (P1) or null (P0).  Pass `encoder` to record into an existing encoder (then nothing is submitted).
   */
  density(positions, weights, ne, { n = Math.round(Math.sqrt(positions.size / 8)), encoder = null, maxb = EXACT2D_MAXB } = {}) {
    const dev = this.device, N = n * n, M = ne * ne;
    const own = !encoder;
    const enc = encoder || dev.createCommandEncoder({ label: 'exact2d' });
    const acc = this._buf('acc', 4 * M), rho = this._buf('rho', 4 * M), stats = this._buf('stats', 16);
    enc.clearBuffer(acc);
    enc.clearBuffer(stats);
    const slot = this.ring.write([['u', N], ['u', n], ['u', ne], ['u', EXACT2D_SCALE], ['u', weights ? 1 : 0], ['u', maxb], ['u', 0], ['u', 0], ne, ne / n]);
    const bg = makeBindGroup(dev, this.pipe.bgl, [this.ring.resource(48), positions, acc, weights || this.dummy, stats], 'exact2d bg');
    const pass = enc.beginComputePass({ label: 'exact triangles' });
    pass.setPipeline(this.pipe.pipeline);
    pass.setBindGroup(0, bg, [slot]);
    dispatch1d(pass, N, 64);
    pass.end();
    const slot2 = this.ring.write([['u', M], ['u', EXACT2D_SCALE]]);
    const bg2 = makeBindGroup(dev, this.conv.bgl, [this.ring.resource(16), acc, rho], 'exact2d conv bg');
    const pass2 = enc.beginComputePass({ label: 'exact convert' });
    pass2.setPipeline(this.conv.pipeline);
    pass2.setBindGroup(0, bg2, [slot2]);
    dispatch1d(pass2, M, 64);
    pass2.end();
    if (own) dev.queue.submit([enc.finish()]);
    return rho;
  }
  /** Counters of the last deposit: {skippedBox: triangles with a bounding box above MAXB cells, skippedDegenerate}. */
  async stats() {
    const s = await readU32(this.device, this._buf('stats', 16), 4);
    return { skippedBox: s[0], skippedDegenerate: s[2] };
  }
}

const helperCache = new WeakMap();
/** One-shot helper: positions / weights as Float32Array or GPUBuffer; returns the ne*ne density as a GPUBuffer (owned by a per-device GpuExact2D). */
export async function exactDensity2D(device, positions, weights, ne) {
  let ex = helperCache.get(device);
  if (!ex) { ex = await new GpuExact2D(device).init(); helperCache.set(device, ex); }
  const tmp = [];
  const up = (a) => {
    if (!(a instanceof Float32Array)) return a;
    const b = device.createBuffer({ size: Math.max(16, a.byteLength), usage: STORAGE_RW });
    device.queue.writeBuffer(b, 0, a);
    tmp.push(b);
    return b;
  };
  const pb = up(positions), wb = weights ? up(weights) : null;
  const rho = ex.density(pb, wb, ne, { n: Math.round(Math.sqrt(pb.size / 8)) });
  for (const b of tmp) device.queue.onSubmittedWorkDone().then(() => b.destroy());
  return rho;
}

// ---------------------------------------------------------------------------------------------------------------------------
// fixtures

const fixCache = new WeakMap();

/**
 * Run the GPU triangle clipper on fixture triangles: cases [{v: Float64Array(6) absolute vertices, d: cell size}].  Returns per case
 * a Map "i,j" -> [m0, mx, my] of ABSOLUTE moments (the kernel orients the triangle counter-clockwise).
 */
export async function exactFixtures2D(device, cases, maxb = 48) {
  let st = fixCache.get(device);
  if (!st) {
    st = (async () => {
      const m = await compile(device, WGSL_FIX, 'exact2d fixtures');
      return makePipeline(device, m, 'fix_tri', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RW_ENTRY(3)], 'fix_tri');
    })();
    fixCache.set(device, st);
  }
  const pipe = await st;
  const ring = ParamRing.get(device);
  const nc = cases.length, maxrec = Math.max(4096, nc * 4096);
  const input = new Float32Array(6 * nc);
  const bases = [];
  cases.forEach((c, i) => {
    const d = c.d;
    const B = [0, 1].map((a) => Math.floor(Math.min(c.v[a], c.v[2 + a], c.v[4 + a]) / d));
    bases.push(B);
    for (let k = 0; k < 3; k++) for (let a = 0; a < 2; a++) input[6 * i + 2 * k + a] = c.v[2 * k + a] / d - B[a];
  });
  const bIn = device.createBuffer({ size: Math.max(16, input.byteLength), usage: STORAGE_RW });
  const bOut = device.createBuffer({ size: 32 * maxrec, usage: STORAGE_RW });
  const bCnt = device.createBuffer({ size: 16, usage: STORAGE_RW });
  device.queue.writeBuffer(bIn, 0, input);
  device.queue.writeBuffer(bCnt, 0, new Uint32Array(4));
  const slot = ring.write([['u', nc], ['u', maxb], ['u', maxrec]]);
  const bg = makeBindGroup(device, pipe.bgl, [ring.resource(16), bIn, bOut, bCnt], 'fix2 bg');
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe.pipeline); pass.setBindGroup(0, bg, [slot]); dispatch1d(pass, nc, 32); pass.end();
  device.queue.submit([enc.finish()]);
  const cnt = await readU32(device, bCnt, 4);
  const nrec = Math.min(cnt[0], maxrec);
  const rec = new Float32Array(await (async () => {
    const stg = device.createBuffer({ size: Math.max(16, 32 * nrec), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    if (nrec) e.copyBufferToBuffer(bOut, 0, stg, 0, 32 * nrec);
    device.queue.submit([e.finish()]);
    await stg.mapAsync(GPUMapMode.READ);
    const ab = stg.getMappedRange().slice(0);
    stg.unmap(); stg.destroy();
    return ab;
  })());
  bIn.destroy(); bOut.destroy(); bCnt.destroy();
  const out = cases.map(() => new Map());
  for (let r = 0; r < nrec; r++) {
    const o = 8 * r, ci = rec[o];
    const B = bases[ci], d = cases[ci].d;
    const cell = [rec[o + 1] + B[0], rec[o + 2] + B[1]];
    const m0 = rec[o + 4], m1 = [rec[o + 5], rec[o + 6]];
    const d2 = d * d;
    out[ci].set(cell.join(','), [d2 * m0, ...[0, 1].map((a) => d2 * d * (cell[a] * m0 + m1[a]))]);
  }
  return { cells: out, overflowBox: cnt[1], records: cnt[0] };
}
