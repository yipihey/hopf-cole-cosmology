// exact3d.js - exact (conservative, clipping-based) simplex deposition in 3-D: WGSL port of the r3d tetrahedron clipper.
//
// A tetrahedron is stored as a vertex graph (positions + 3 neighbour indices per vertex, ordered so that walking (np + 1) % 3
// turns around a face), exactly as in core/src/r3d.rs::Poly<3>.  For each cell of the tetrahedron's bounding box the tetrahedron is
// clipped against the six axis-aligned cell planes (one-sided clip = r3d_clip against a plane: new vertices on the cut edges,
// face-walk linking with (np + 1) % 3, compaction), nested slab by slab (x-slab, then y-slab of that, then z-slab) so that the
// outer clips are shared by all the cells behind them, and the moments of order <= 1 (volume m0 and first moments m1) of the clipped
// polytope are summed over its faces (Koehl recursion for order 1: sixv/6 and sixv (v0 + v1 + v2)/24 per face-fan triangle).
//
// All coordinates are in cell units RELATIVE TO A BASE CELL (an integer offset the caller adds back), and the moments are taken
// about the lower corner of each cell, so every number is O(1) and the f32 result agrees with the f64 reference to ~1e-6.
//
// The polytope vertices also carry one scalar attribute (phi, interpolated linearly along the cut edges), so a P1 density profile is
// integrated exactly without ever being extrapolated: integ() returns (volume, int phi).
//
// The shared source (WGSL_CLIP3D) leaves two things to the including shader:  fn emit(cell: vec3<i32>, p: ptr<private, Poly>)  (called for
// every non-empty cell piece) and the caller's own use of P_T / walk_tet().

import { makePipeline, makeBindGroup, dispatch1d, compile, ParamRing, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';

export const WGSL_CLIP3D = /* wgsl */`
const CAP: u32 = 40u;
const NIL: u32 = 255u;
struct Poly { pos: array<vec3<f32>, CAP>, phi: array<f32, CAP>, nbr: array<u32, CAP>, n: u32 };   // phi: scalar attribute per vertex (linear inside)

var<private> P_T: Poly;      // the tetrahedron
var<private> P_X: Poly;      // ... clipped to the x-slab
var<private> P_Y: Poly;      // ... and the y-slab
var<private> P_A: Poly;      // scratch (second half of every slab clip)
var<private> P_B: Poly;      // the cell piece
var<private> sdv: array<f32, CAP>;
var<private> idxv: array<u32, CAP>;
var<private> emk: array<u32, CAP>;
var<private> g_ovf: u32;

fn pack3(a: u32, b: u32, c: u32) -> u32 { return a | (b << 8u) | (c << 16u); }
fn nbget(p: ptr<private, Poly>, v: u32, k: u32) -> u32 { return ((*p).nbr[v] >> (8u * k)) & 255u; }

fn init_tet(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, f: vec4<f32>) {   // positively oriented: (b-a, c-a, d-a) right handed
  P_T.pos[0] = a; P_T.pos[1] = b; P_T.pos[2] = c; P_T.pos[3] = d;
  P_T.phi[0] = f.x; P_T.phi[1] = f.y; P_T.phi[2] = f.z; P_T.phi[3] = f.w;
  P_T.nbr[0] = pack3(1u, 3u, 2u); P_T.nbr[1] = pack3(2u, 3u, 0u); P_T.nbr[2] = pack3(0u, 3u, 1u); P_T.nbr[3] = pack3(1u, 2u, 0u);
  P_T.n = 4u;
}

// One-sided clip of src into dst: keep sgn * (x[ax] - coord) <= 0.  Returns 0 (empty), 1 (nothing clipped, dst = copy of src) or 2.
fn clip(src: ptr<private, Poly>, dst: ptr<private, Poly>, ax: u32, coord: f32, sgn: f32) -> u32 {
  let onv = (*src).n;
  var nin = 0u;
  for (var v = 0u; v < onv; v++) {
    let s = sgn * ((*src).pos[v][ax] - coord);
    sdv[v] = s;
    if (s > 0.0) { idxv[v] = NIL; } else { idxv[v] = nin; nin++; }
  }
  if (nin == 0u) { return 0u; }
  if (nin == onv) {
    for (var v = 0u; v < onv; v++) { (*dst).pos[v] = (*src).pos[v]; (*dst).phi[v] = (*src).phi[v]; (*dst).nbr[v] = (*src).nbr[v]; }
    (*dst).n = onv;
    return 1u;
  }
  var nn = nin;
  for (var v = 0u; v < onv; v++) {
    let iv = idxv[v];
    if (iv == NIL) { continue; }
    var packed = 0u;
    for (var np = 0u; np < 3u; np++) {
      let w = nbget(src, v, np);
      var tgt = idxv[w];
      if (tgt == NIL) {
        if (nn >= CAP) { g_ovf = 1u; return 0u; }
        let wa = -sdv[w];
        let wb = sdv[v];
        let inv = 1.0 / (wa + wb);
        (*dst).pos[nn] = (wa * (*src).pos[v] + wb * (*src).pos[w]) * inv;
        (*dst).phi[nn] = (wa * (*src).phi[v] + wb * (*src).phi[w]) * inv;
        (*dst).nbr[nn] = pack3(iv, NIL, NIL);
        tgt = nn; nn++;
      }
      packed = packed | (tgt << (8u * np));
    }
    (*dst).pos[iv] = (*src).pos[v];
    (*dst).phi[iv] = (*src).phi[v];
    (*dst).nbr[iv] = packed;
  }
  // link the new vertices around the cut face: walk (np + 1) % 3 from each until the next new vertex
  for (var vs = nin; vs < nn; vs++) {
    var vc = vs;
    var vn = nbget(dst, vc, 0u);
    for (var it = 0u; it < 64u; it++) {
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(dst, vn, k) == vc) { np = k; break; } }
      vc = vn;
      vn = nbget(dst, vc, (np + 1u) % 3u);
      if (vc >= nin) { break; }
    }
    (*dst).nbr[vs] = ((*dst).nbr[vs] & ~(255u << 16u)) | (vc << 16u);
    (*dst).nbr[vc] = ((*dst).nbr[vc] & ~(255u << 8u)) | (vs << 8u);
  }
  (*dst).n = nn;
  return 2u;
}

// Clip to lo <= x[ax] <= hi (through the scratch polytope P_A).
fn slab(src: ptr<private, Poly>, dst: ptr<private, Poly>, ax: u32, lo: f32, hi: f32) -> u32 {
  if (clip(src, &P_A, ax, lo, -1.0) == 0u) { return 0u; }
  return clip(&P_A, dst, ax, hi, 1.0);
}

// (volume, first moments) about 'org' by fanning every face (Koehl recursion, order 1)
fn moments1(p: ptr<private, Poly>, org: vec3<f32>) -> vec4<f32> {
  let nv = (*p).n;
  for (var v = 0u; v < nv; v++) { emk[v] = 0u; }
  var m0 = 0.0;
  var m1 = vec3<f32>(0.0);
  for (var vstart = 0u; vstart < nv; vstart++) {
    for (var pstart = 0u; pstart < 3u; pstart++) {
      if ((emk[vstart] & (1u << pstart)) != 0u) { continue; }
      var vcur = vstart;
      var pnext = pstart;
      emk[vcur] = emk[vcur] | (1u << pnext);
      var vnext = nbget(p, vcur, pnext);
      let v0 = (*p).pos[vcur] - org;
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
      vcur = vnext;
      pnext = (np + 1u) % 3u;
      emk[vcur] = emk[vcur] | (1u << pnext);
      vnext = nbget(p, vcur, pnext);
      for (var it = 0u; it < 64u; it++) {
        if (vnext == vstart) { break; }
        let v2 = (*p).pos[vcur] - org;
        let v1 = (*p).pos[vnext] - org;
        let sixv = dot(v0, cross(v1, v2));
        m0 += sixv;
        m1 += sixv * (v0 + v1 + v2);
        np = 0u;
        for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
        vcur = vnext;
        pnext = (np + 1u) % 3u;
        emk[vcur] = emk[vcur] | (1u << pnext);
        vnext = nbget(p, vcur, pnext);
      }
    }
  }
  return vec4<f32>(m0 / 6.0, m1 / 24.0);
}


// (volume, integral of the linear scalar phi) of the polytope, fanned from its vertex 0 so that phi is only interpolated, never
// extrapolated (the P1 shape of a sliver tetrahedron has large barycentric gradients: evaluating it at a distant origin would cancel)
fn integ(p: ptr<private, Poly>) -> vec2<f32> {
  let nv = (*p).n;
  for (var v = 0u; v < nv; v++) { emk[v] = 0u; }
  let o = (*p).pos[0];
  let f0 = (*p).phi[0];
  var vol = 0.0;
  var ip = 0.0;
  for (var vstart = 0u; vstart < nv; vstart++) {
    for (var pstart = 0u; pstart < 3u; pstart++) {
      if ((emk[vstart] & (1u << pstart)) != 0u) { continue; }
      var vcur = vstart;
      var pnext = pstart;
      emk[vcur] = emk[vcur] | (1u << pnext);
      var vnext = nbget(p, vcur, pnext);
      let v0 = (*p).pos[vstart] - o;
      let g0 = (*p).phi[vstart];
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
      vcur = vnext;
      pnext = (np + 1u) % 3u;
      emk[vcur] = emk[vcur] | (1u << pnext);
      vnext = nbget(p, vcur, pnext);
      for (var it = 0u; it < 64u; it++) {
        if (vnext == vstart) { break; }
        let v2 = (*p).pos[vcur] - o;
        let v1 = (*p).pos[vnext] - o;
        let sixv = dot(v0, cross(v1, v2));
        vol += sixv;
        ip += sixv * (f0 + g0 + (*p).phi[vcur] + (*p).phi[vnext]);
        np = 0u;
        for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
        vcur = vnext;
        pnext = (np + 1u) % 3u;
        emk[vcur] = emk[vcur] | (1u << pnext);
        vnext = nbget(p, vcur, pnext);
      }
    }
  }
  return vec2<f32>(vol / 6.0, ip / 24.0);
}

// Visit the cells [blo, bhi) (relative to the base cell) overlapped by the tetrahedron P_T and emit their moments.
fn walk_tet(blo: vec3<i32>, bhi: vec3<i32>) {
  for (var ci = blo.x; ci < bhi.x; ci++) {
    if (slab(&P_T, &P_X, 0u, f32(ci), f32(ci + 1)) == 0u) { continue; }
    var ylo = P_X.pos[0].y; var yhi = ylo;
    for (var v = 1u; v < P_X.n; v++) { ylo = min(ylo, P_X.pos[v].y); yhi = max(yhi, P_X.pos[v].y); }
    let j0 = max(blo.y, i32(floor(ylo)));
    let j1 = min(bhi.y, i32(ceil(yhi)));
    for (var cj = j0; cj < j1; cj++) {
      if (slab(&P_X, &P_Y, 1u, f32(cj), f32(cj + 1)) == 0u) { continue; }
      var zlo = P_Y.pos[0].z; var zhi = zlo;
      for (var v = 1u; v < P_Y.n; v++) { zlo = min(zlo, P_Y.pos[v].z); zhi = max(zhi, P_Y.pos[v].z); }
      let k0 = max(blo.z, i32(floor(zlo)));
      let k1 = min(bhi.z, i32(ceil(zhi)));
      for (var ck = k0; ck < k1; ck++) {
        if (slab(&P_Y, &P_B, 2u, f32(ck), f32(ck + 1)) == 0u) { continue; }
        emit(vec3<i32>(ci, cj, ck), &P_B);
      }
    }
  }
}
`;

// ---------------------------------------------------------------------------------------------------------------------------
// fixture kernel: one thread per tetrahedron, appends (case, cell, m0, m1) records

const WGSL_FIX = /* wgsl */`
${WGSL_LINEAR}
${WGSL_CLIP3D}
struct FU { ncases: u32, maxb: u32, maxrec: u32 };
@group(0) @binding(0) var<uniform> fu: FU;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read_write> fcnt: array<atomic<u32>>;
var<private> c_case: u32;

fn emit(cell: vec3<i32>, p: ptr<private, Poly>) {
  let m = moments1(p, vec3<f32>(cell));
  let r = atomicAdd(&fcnt[0], 1u);
  if (r >= fu.maxrec) { return; }
  let o = 8u * r;
  fout[o] = f32(c_case); fout[o + 1u] = f32(cell.x); fout[o + 2u] = f32(cell.y); fout[o + 3u] = f32(cell.z);
  fout[o + 4u] = m.x; fout[o + 5u] = m.y; fout[o + 6u] = m.z; fout[o + 7u] = m.w;
}

@compute @workgroup_size(32)
fn fix_tet(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 32u);
  if (i >= fu.ncases) { return; }
  c_case = i;
  var v: array<vec3<f32>, 4>;
  for (var k = 0u; k < 4u; k++) { v[k] = vec3<f32>(fin[12u * i + 3u * k], fin[12u * i + 3u * k + 1u], fin[12u * i + 3u * k + 2u]); }
  let det = dot(v[1] - v[0], cross(v[2] - v[0], v[3] - v[0]));
  if (det < 0.0) { let t = v[1]; v[1] = v[2]; v[2] = t; }
  init_tet(v[0], v[1], v[2], v[3], vec4<f32>(1.0));
  var lo = v[0]; var hi = v[0];
  for (var k = 1u; k < 4u; k++) { lo = min(lo, v[k]); hi = max(hi, v[k]); }
  let blo = vec3<i32>(floor(lo));
  var bhi = vec3<i32>(ceil(hi));
  bhi = max(bhi, blo + vec3<i32>(1));
  let ext = bhi - blo;
  if (max(ext.x, max(ext.y, ext.z)) > i32(fu.maxb)) { atomicAdd(&fcnt[1], 1u); return; }
  walk_tet(blo, bhi);
  if (g_ovf != 0u) { atomicAdd(&fcnt[2], 1u); }
}
`;

const fixCache = new WeakMap();

/**
 * Run the GPU tetrahedron clipper on fixture tetrahedra.
 * cases: [{v: Float64Array(12) absolute vertex coordinates, d: cell size}].  Each tetrahedron is converted to cell units and
 * shifted by the integer base cell B = floor(min / d) in f64 (the f32 kernel works on the O(1) remainder).
 * Returns per case a Map "i,j,k" -> [m0, mx, my, mz] with the ABSOLUTE moments (int x^a dV, a <= 1) in the input orientation's
 * sign convention left to the caller (the kernel orients the tetrahedron positively), plus the overflow counters.
 */
export async function exactFixtures3D(device, cases, maxb = 48) {
  let st = fixCache.get(device);
  if (!st) {
    st = (async () => {
      const m = await compile(device, WGSL_FIX, 'exact3d fixtures');
      return makePipeline(device, m, 'fix_tet', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RW_ENTRY(3)], 'fix_tet');
    })();
    fixCache.set(device, st);
  }
  const pipe = await st;
  const ring = ParamRing.get(device);
  const nc = cases.length, maxrec = Math.max(4096, nc * 4096);
  const input = new Float32Array(12 * nc);
  const bases = [];
  cases.forEach((c, i) => {
    const d = c.d;
    const B = [0, 1, 2].map((a) => Math.floor(Math.min(c.v[a], c.v[3 + a], c.v[6 + a], c.v[9 + a]) / d));
    bases.push(B);
    for (let k = 0; k < 4; k++) for (let a = 0; a < 3; a++) input[12 * i + 3 * k + a] = c.v[3 * k + a] / d - B[a];
  });
  const bIn = device.createBuffer({ size: Math.max(16, input.byteLength), usage: STORAGE_RW });
  const bOut = device.createBuffer({ size: 32 * maxrec, usage: STORAGE_RW });
  const bCnt = device.createBuffer({ size: 16, usage: STORAGE_RW });
  device.queue.writeBuffer(bIn, 0, input);
  device.queue.writeBuffer(bCnt, 0, new Uint32Array(4));
  const slot = ring.write([['u', nc], ['u', maxb], ['u', maxrec]]);
  const bg = makeBindGroup(device, pipe.bgl, [ring.resource(16), bIn, bOut, bCnt], 'fix bg');
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe.pipeline); pass.setBindGroup(0, bg, [slot]); dispatch1d(pass, nc, 32); pass.end();
  const stg1 = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  enc.copyBufferToBuffer(bCnt, 0, stg1, 0, 16);
  device.queue.submit([enc.finish()]);
  await stg1.mapAsync(GPUMapMode.READ);
  const cnt = new Uint32Array(stg1.getMappedRange().slice(0));
  stg1.unmap(); stg1.destroy();
  const nrec = Math.min(cnt[0], maxrec);
  const stg2 = device.createBuffer({ size: Math.max(16, 32 * nrec), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc2 = device.createCommandEncoder();
  if (nrec) enc2.copyBufferToBuffer(bOut, 0, stg2, 0, 32 * nrec);
  device.queue.submit([enc2.finish()]);
  await stg2.mapAsync(GPUMapMode.READ);
  const rec = new Float32Array(stg2.getMappedRange().slice(0));
  stg2.unmap(); stg2.destroy(); bIn.destroy(); bOut.destroy(); bCnt.destroy();
  const out = cases.map(() => new Map());
  for (let r = 0; r < nrec; r++) {
    const o = 8 * r, ci = rec[o];
    const B = bases[ci], d = cases[ci].d;
    const cell = [rec[o + 1] + B[0], rec[o + 2] + B[1], rec[o + 3] + B[2]];
    const m0 = rec[o + 4], m1 = [rec[o + 5], rec[o + 6], rec[o + 7]];
    const d3 = d * d * d;
    out[ci].set(cell.join(','), [d3 * m0, ...[0, 1, 2].map((a) => d3 * d * (cell[a] * m0 + m1[a]))]);
  }
  return { cells: out, overflowBox: cnt[1], overflowCap: cnt[2], records: cnt[0] };
}
