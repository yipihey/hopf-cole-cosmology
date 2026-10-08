// sheet3d.js - phase-space-sheet density of the 3-D lab on the GPU (same quantity as sheet.rs::sheet_density_3d, ss = 1).
//
//   const sheet = new GpuSheet3D(g); await sheet.init();
//   rho = sheet.density(dispBuffer [, weightsBuffer|null [, encoder]])
//                                                    GPUBuffer n^3 f32, rho/rhobar; dispBuffer = Psi = x - q (box units, interleaved xyz,
//                                                    e.g. g.displacement(gvals, order)); persistent buffer 'sheet-rho' of g
//   w   = sheet.vertexWeights(dispBuffer [, encoder])  GPUBuffer n^3 f32, w_v = 1/max(|J(q_v)|, 1e-4) on the Lagrangian grid (= WASM
//                                                    vertex_density up to the 4th-order finite-difference Jacobian), buffer 'sheet-w'
//
//   rho = sheet.densityExact(dispBuffer [, weightsBuffer|null [, encoder]])
//                                                    GPUBuffer n^3 f32 (persistent buffer 'sheetx-rho' of g): the EXACT (conservative) deposit of
//                                                    the same tetrahedra: every tetrahedron is clipped against the Eulerian cells it overlaps and
//                                                    deposits the exact integral of its P0 (constant) or P1 (weights given) density profile, see
//                                                    exact3d.js; stats() = {skippedBox, skippedCap} (tetrahedra over the bounding-box clamp / over the
//                                                    vertex cap of the clipper, whose mass is lost)
//
// P1 sheet: with a weights buffer (vertex densities, index ix*n^2+iy*n+iz) every sample inside a tetrahedron receives
//   (m/|V|) (sum_i lambda_i w_i) / mean_i(w_i)   (lambda = barycentric coordinates of the sample point)
// instead of m/|V|: a linear density shape per simplex that still deposits exactly the simplex mass (sheet.rs::sheet_density_3d_weighted).
//
// One thread per Lagrangian cell.  The cell is split into the six Kuhn tetrahedra (same vertex tables as the CPU:
// [0,1,3,7] [0,1,5,7] [0,2,3,7] [0,2,6,7] [0,4,5,7] [0,4,6,7], corner bits (x,y,z); the periodic +L offsets of the CPU are the
// integer lattice offsets here, because vertices are the lattice point plus Psi of the wrapped index).  For every tetrahedron
// the cells whose centre may lie inside it (bounding box) are visited and tested; a covered cell centre receives m/|V| = 1/|vol6|
// (cell units) as an 18-bit fixed-point integer atomicAdd (2^18 per unit rho/rhobar, wrapping above rho/rhobar = 16384, exactly the
// CIC convention).
//
// Watertight point test (this is what makes the result agree with the f64 CPU raster to ~1e-6 instead of ~1e-3):
//  * vertex coordinates relative to the SAMPLE POINT, w_v = (lattice_v - cell) - 1/2 + n Psi_v: the first term is a small exact
//    integer/half-integer and n Psi_v is bit-identical for the same vertex, so two tetrahedra sharing a vertex (also across the
//    periodic seam) see bit-identical w_v;
//  * the face orientation det[w_b - w_a, w_c - w_a, -w_a] of a face is evaluated with its three vertices sorted by global vertex
//    id, so the two tetrahedra sharing a face get the same number and the point falls into exactly one of them (a point with
//    determinant 0 counts as positive, consistently);
//  * inside = all four faces have the sign of the opposite vertex.
// Bounding boxes are limited to n cells per axis (a tetrahedron stretched over more than the box would double count itself).

import { readU32, ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';
import { CIC_SCALE } from './cosmo3d.js';
import { WGSL_CLIP3D } from './exact3d.js';

export const EXACT3D_MAXB = 12;      // largest bounding box (cells per axis) a tetrahedron may span in the exact deposit

const WGSL = /* wgsl */`
${WGSL_LINEAR}
struct SU { count: u32, n: u32, scale: u32, useW: u32, nf: f32 };
@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> wv: array<f32>;

var<private> TETS: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
  vec4<u32>(0u, 1u, 3u, 7u), vec4<u32>(0u, 1u, 5u, 7u), vec4<u32>(0u, 2u, 3u, 7u),
  vec4<u32>(0u, 2u, 6u, 7u), vec4<u32>(0u, 4u, 5u, 7u), vec4<u32>(0u, 4u, 6u, 7u));

fn vtx(b: u32, ix: i32, iy: i32, iz: i32, n: u32) -> u32 {
  let mask = n - 1u;
  return ((u32(ix + i32(b & 1u)) & mask) * n + (u32(iy + i32((b >> 1u) & 1u)) & mask)) * n + (u32(iz + i32((b >> 2u) & 1u)) & mask);
}

fn orient(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  return dot(cross(b - a, c - a), -a);
}

@compute @workgroup_size(64)
fn sheet_tet(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 64u);
  if (i >= su.count) { return; }
  let n = su.n;
  let mask = n - 1u;
  let ix = i32(i / (n * n)); let iy = i32((i / n) % n); let iz = i32(i % n);
  let nf = su.nf;
  var lc: array<vec3<f32>, 8>;     // cell-local vertex coordinate: corner bits + n Psi
  var pw: array<vec3<f32>, 8>;     // n Psi
  for (var b = 0u; b < 8u; b = b + 1u) {
    let bx = b & 1u; let by = (b >> 1u) & 1u; let bz = (b >> 2u) & 1u;
    let idx = vtx(b, ix, iy, iz, n);
    let p = vec3<f32>(psi[3u * idx], psi[3u * idx + 1u], psi[3u * idx + 2u]) * nf;
    pw[b] = p;
    lc[b] = vec3<f32>(f32(bx), f32(by), f32(bz)) + p;
  }
  for (var t = 0u; t < 6u; t = t + 1u) {
    let tv = TETS[t];
    var vb = array<u32, 4>(tv.x, tv.y, tv.z, tv.w);
    var l4: array<vec3<f32>, 4>; var p4: array<vec3<f32>, 4>; var bits: array<vec3<i32>, 4>; var id4: array<u32, 4>;
    for (var k = 0u; k < 4u; k = k + 1u) {
      let b = vb[k];
      l4[k] = lc[b]; p4[k] = pw[b]; id4[k] = vtx(b, ix, iy, iz, n);
      bits[k] = vec3<i32>(i32(b & 1u), i32((b >> 1u) & 1u), i32((b >> 2u) & 1u));
    }
    let vol6 = dot(l4[1] - l4[0], cross(l4[2] - l4[0], l4[3] - l4[0]));
    if (abs(vol6) < 1e-9) { continue; }
    let dens = 1.0 / abs(vol6);
    let q = u32(min(dens * f32(su.scale) + 0.5, 2147483648.0));
    // P1 shape: vertex weights of this tetrahedron and their mean (flat shape when there is no weight buffer or the mean is not positive)
    var w4 = vec4<f32>(1.0);
    var wmean = 1.0;
    if (su.useW != 0u) {
      w4 = vec4<f32>(wv[id4[0]], wv[id4[1]], wv[id4[2]], wv[id4[3]]);
      let m = 0.25 * (w4.x + w4.y + w4.z + w4.w);
      if (m > 0.0) { wmean = m; } else { w4 = vec4<f32>(1.0); }
    }
    let dscale = dens * f32(su.scale) / wmean;
    let tinv = 1.0 / vol6;
    // faces: opposite vertex f, the other three sorted by vertex id, sign of the opposite vertex
    var fa: array<u32, 4>; var fb: array<u32, 4>; var fc: array<u32, 4>; var sv: array<f32, 4>;
    for (var f = 0u; f < 4u; f = f + 1u) {
      var s0 = 0u; var s1 = 1u; var s2 = 2u;
      if (f == 0u) { s0 = 1u; s1 = 2u; s2 = 3u; } else if (f == 1u) { s0 = 0u; s1 = 2u; s2 = 3u; } else if (f == 2u) { s0 = 0u; s1 = 1u; s2 = 3u; }
      var tmp = 0u;
      if (id4[s0] > id4[s1]) { tmp = s0; s0 = s1; s1 = tmp; }
      if (id4[s1] > id4[s2]) { tmp = s1; s1 = s2; s2 = tmp; }
      if (id4[s0] > id4[s1]) { tmp = s0; s0 = s1; s1 = tmp; }
      fa[f] = s0; fb[f] = s1; fc[f] = s2;
      let d = dot(cross(l4[s1] - l4[s0], l4[s2] - l4[s0]), l4[f] - l4[s0]);
      sv[f] = select(-1.0, 1.0, d >= 0.0);
    }
    // bounding box in cell units (absolute), sample centres at integer + 1/2
    var lo = l4[0]; var hi = l4[0];
    for (var k = 1u; k < 4u; k = k + 1u) { lo = min(lo, l4[k]); hi = max(hi, l4[k]); }
    let org = vec3<f32>(f32(ix), f32(iy), f32(iz));
    let c0 = vec3<i32>(ceil(lo + org - vec3<f32>(0.501)));
    var c1 = vec3<i32>(floor(hi + org - vec3<f32>(0.499)));
    c1 = min(c1, c0 + vec3<i32>(i32(n) - 1));
    for (var ci = c0.x; ci <= c1.x; ci = ci + 1) {
      for (var cj = c0.y; cj <= c1.y; cj = cj + 1) {
        for (var ck = c0.z; ck <= c1.z; ck = ck + 1) {
          let d = vec3<i32>(ci - ix, cj - iy, ck - iz);
          var w: array<vec3<f32>, 4>;
          for (var k = 0u; k < 4u; k = k + 1u) {
            let o = bits[k] - d;
            w[k] = vec3<f32>(f32(o.x) - 0.5, f32(o.y) - 0.5, f32(o.z) - 0.5) + p4[k];
          }
          var inside = true;
          for (var f = 0u; f < 4u; f = f + 1u) {
            let sp = orient(w[fa[f]], w[fb[f]], w[fc[f]]);
            if (select(-1.0, 1.0, sp >= 0.0) != sv[f]) { inside = false; break; }
          }
          if (inside) {
            let idx = ((u32(ci) & mask) * n + (u32(cj) & mask)) * n + (u32(ck) & mask);
            if (su.useW != 0u) {
              // barycentric coordinates of the sample point (the origin of the w coordinates)
              let e1 = w[1] - w[0]; let e2 = w[2] - w[0]; let e3 = w[3] - w[0];
              let l1 = dot(-w[0], cross(e2, e3)) * tinv;
              let l2 = dot(e1, cross(-w[0], e3)) * tinv;
              let l3 = dot(e1, cross(e2, -w[0])) * tinv;
              let l0 = 1.0 - l1 - l2 - l3;
              let fw = max(l0 * w4.x + l1 * w4.y + l2 * w4.z + l3 * w4.w, 0.0);
              atomicAdd(&acc[idx], u32(min(fw * dscale + 0.5, 2147483648.0)));
            } else {
              atomicAdd(&acc[idx], q);
            }
          }
        }
      }
    }
  }
}
`;


// Exact deposit: one thread per Lagrangian cell, six Kuhn tetrahedra; every tetrahedron is clipped against the cells of its bounding
// box (exact3d.js).  Coordinates are cell units of the Eulerian grid relative to the Lagrangian cell's own base cell, so they are O(1)
// numbers even at large displacement.
const WGSL_EXACT = /* wgsl */`
${WGSL_LINEAR}
${WGSL_CLIP3D}
struct XU { count: u32, n: u32, ne: u32, scale: u32, useW: u32, maxb: u32, base: u32, pad: u32, nef: f32, rat: f32 };
@group(0) @binding(0) var<uniform> xu: XU;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> wv: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

var<private> TETS: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
  vec4<u32>(0u, 1u, 3u, 7u), vec4<u32>(0u, 1u, 5u, 7u), vec4<u32>(0u, 2u, 3u, 7u),
  vec4<u32>(0u, 2u, 6u, 7u), vec4<u32>(0u, 4u, 5u, 7u), vec4<u32>(0u, 4u, 6u, 7u));

var<private> c_base: vec3<i32>;
var<private> c_dens: f32;

fn wrapi(c: i32, ne: i32) -> u32 { return u32(((c % ne) + ne) % ne); }

fn emit(cell: vec3<i32>, p: ptr<private, Poly>) {
  // (volume, int phi) of the piece, phi = (sum_c w_c lambda_c) / mean(w) carried (and interpolated) by the polytope vertices
  let r = integ(p);
  if (r.x <= 1e-12) { return; }
  let q = u32(min(c_dens * max(r.y, 0.0) * f32(xu.scale) + 0.5, 2147483648.0));
  let ne = i32(xu.ne);
  let c = c_base + cell;
  atomicAdd(&acc[(wrapi(c.x, ne) * xu.ne + wrapi(c.y, ne)) * xu.ne + wrapi(c.z, ne)], q);
}

@compute @workgroup_size(32)
fn exact_tet(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 32u) + xu.base;
  if (i >= xu.count) { return; }
  let n = xu.n;
  let ix = i / (n * n); let iy = (i / n) % n; let iz = i % n;
  let org = vec3<f32>(f32(ix), f32(iy), f32(iz)) * xu.rat;       // lower corner of the Lagrangian cell in Eulerian cell units
  let bf = floor(org);
  c_base = vec3<i32>(bf);
  let off = org - bf;
  var cp: array<vec3<f32>, 8>;
  var cw: array<f32, 8>;
  for (var b = 0u; b < 8u; b++) {
    let jx = (ix + (b & 1u)) % n; let jy = (iy + ((b >> 1u) & 1u)) % n; let jz = (iz + ((b >> 2u) & 1u)) % n;
    let idx = (jx * n + jy) * n + jz;
    let bits = vec3<f32>(f32(b & 1u), f32((b >> 1u) & 1u), f32((b >> 2u) & 1u));
    cp[b] = off + bits * xu.rat + vec3<f32>(psi[3u * idx], psi[3u * idx + 1u], psi[3u * idx + 2u]) * xu.nef;
    cw[b] = select(1.0, wv[idx], xu.useW != 0u);
  }
  for (var t = 0u; t < 6u; t++) {
    let tv = TETS[t];
    var v0 = cp[tv.x]; var v1 = cp[tv.y]; var v2 = cp[tv.z]; var v3 = cp[tv.w];
    var w0 = cw[tv.x]; var w1 = cw[tv.y]; var w2 = cw[tv.z]; var w3 = cw[tv.w];
    var e1 = v1 - v0; var e2 = v2 - v0; var e3 = v3 - v0;
    var det = dot(e1, cross(e2, e3));
    if (abs(det) < 1e-9) { atomicAdd(&stats[2], 1u); continue; }
    if (det < 0.0) {
      let tp = v1; v1 = v2; v2 = tp; let tw = w1; w1 = w2; w2 = tw;
      e1 = v1 - v0; e2 = v2 - v0; det = -det;
    }
    let lo = min(min(v0, v1), min(v2, v3)); let hi = max(max(v0, v1), max(v2, v3));
    let blo = vec3<i32>(floor(lo));
    let bhi = max(vec3<i32>(ceil(hi)), blo + vec3<i32>(1));
    let ext = bhi - blo;
    if (max(ext.x, max(ext.y, ext.z)) > i32(xu.maxb)) { atomicAdd(&stats[0], 1u); continue; }
    c_dens = xu.rat * xu.rat * xu.rat / det;
    var phi = vec4<f32>(1.0);
    if (xu.useW != 0u) {
      let wm = 0.25 * (w0 + w1 + w2 + w3);
      if (wm > 0.0) { phi = vec4<f32>(w0, w1, w2, w3) / wm; }
    }
    init_tet(v0, v1, v2, v3, phi);
    g_ovf = 0u;
    walk_tet(blo, bhi);
    if (g_ovf != 0u) { atomicAdd(&stats[1], 1u); }
  }
}
`;

// Vertex densities w_v = 1/max(|J|, 1e-4), J = det(I + grad Psi) at the Lagrangian grid points (4th-order central differences, as in
// lpt3d.js::st_jac; the CPU's vertex_density uses spectral derivatives).
const WGSL_W = /* wgsl */`
${WGSL_LINEAR}
struct WU { count: u32, n: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> wu: WU;
@group(0) @binding(1) var<storage, read> psiw: array<f32>;
@group(0) @binding(2) var<storage, read_write> wout: array<f32>;

fn wrapi(x: i32, n: u32) -> u32 { return u32(x & (i32(n) - 1)); }
fn p3(x: i32, y: i32, z: i32, n: u32) -> vec3<f32> {
  let b = 3u * ((wrapi(x, n) * n + wrapi(y, n)) * n + wrapi(z, n));
  return vec3<f32>(psiw[b], psiw[b + 1u], psiw[b + 2u]);
}

@compute @workgroup_size(64)
fn vertex_w(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 64u);
  if (i >= wu.count) { return; }
  let n = wu.n;
  let x = i32(i / (n * n)); let y = i32((i / n) % n); let z = i32(i % n);
  let s = f32(n) / 12.0;
  let cx = (8.0 * (p3(x + 1, y, z, n) - p3(x - 1, y, z, n)) - (p3(x + 2, y, z, n) - p3(x - 2, y, z, n))) * s;
  let cy = (8.0 * (p3(x, y + 1, z, n) - p3(x, y - 1, z, n)) - (p3(x, y + 2, z, n) - p3(x, y - 2, z, n))) * s;
  let cz = (8.0 * (p3(x, y, z + 1, n) - p3(x, y, z - 1, n)) - (p3(x, y, z + 2, n) - p3(x, y, z - 2, n))) * s;
  let m = mat3x3<f32>(cx + vec3<f32>(1.0, 0.0, 0.0), cy + vec3<f32>(0.0, 1.0, 0.0), cz + vec3<f32>(0.0, 0.0, 1.0));
  wout[i] = 1.0 / max(abs(determinant(m)), 1e-4);
}
`;

const pipeCache = new WeakMap();
function getPipeline(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const m = await compile(device, WGSL, 'sheet3d');
      const mw = await compile(device, WGSL_W, 'sheet3d vertex weights');
      const mx = await compile(device, WGSL_EXACT, 'sheet3d exact');
      return {
        exact: makePipeline(device, mx, 'exact_tet', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RO_ENTRY(3), RW_ENTRY(4)], 'exact_tet'),
        tet: makePipeline(device, m, 'sheet_tet', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2), RO_ENTRY(3)], 'sheet_tet'),
        vw: makePipeline(device, mw, 'vertex_w', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'vertex_w'),
      };
    })();
    pipeCache.set(device, p);
  }
  return p;
}

export class GpuSheet3D {
  constructor(g) { this.g = g; this.device = g.device; this.n = g.n; this.size = g.size; this.ring = ParamRing.get(g.device); }
  async init() {
    const p = await getPipeline(this.device);
    this.pipe = p.tet; this.pipeW = p.vw; this.pipeX = p.exact;
    this.dummy = this.device.createBuffer({ label: 'sheet no weights', size: 16, usage: STORAGE_RW });
    return this;
  }
  destroy() {
    if (this.dummy) { this.dummy.destroy(); this.dummy = null; }
    if (this.xstats) { this.xstats.destroy(); this.xstats = null; }
  }

  /** Vertex densities w_v = 1/max(|J|, 1e-4) of the displacement buffer `disp` (persistent buffer 'sheet-w'). */
  vertexWeights(disp, encoder = null) {
    const g = this.g, N = this.size, n = this.n;
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'vertex weights' });
    const w = g.buf('sheet-w', g.fbytes);
    const slot = this.ring.write([['u', N], ['u', n]]);
    const bg = makeBindGroup(this.device, this.pipeW.bgl, [this.ring.resource(16), disp, w], 'vertex w bg');
    const pass = enc.beginComputePass({ label: 'vertex weights' });
    pass.setPipeline(this.pipeW.pipeline);
    pass.setBindGroup(0, bg, [slot]);
    dispatch1d(pass, N, 64);
    pass.end();
    if (own) g._submit(enc);
    return w;
  }

  /** rho/rhobar of the sheet; `weights` (GPUBuffer of n^3 vertex densities) switches on the P1 shape. */
  density(disp, weights = null, encoder = null) {
    if (weights && typeof weights.beginComputePass === 'function') { encoder = weights; weights = null; }   // old signature density(disp, encoder)
    const g = this.g, N = this.size, n = this.n;
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'sheet' });
    const acc = g.buf('sheet-acc', g.fbytes), rho = g.buf('sheet-rho', g.fbytes);
    enc.clearBuffer(acc);
    const slot = this.ring.write([['u', N], ['u', n], ['u', CIC_SCALE], ['u', weights ? 1 : 0], n]);
    const bg = makeBindGroup(this.device, this.pipe.bgl, [this.ring.resource(32), disp, acc, weights || this.dummy], 'sheet bg');
    const pass = enc.beginComputePass({ label: 'sheet tets' });
    pass.setPipeline(this.pipe.pipeline);
    pass.setBindGroup(0, bg, [slot]);
    dispatch1d(pass, N, 64);
    pass.end();
    const words = [['u', N], ['u', n], ['u', CIC_SCALE], 0, n];
    g._dispatch(enc, 'sheet', g.P.cicConvert, g.bg('sheetconv', g.P.cicConvert, [g.ring.resource(32), acc, rho]), words, N, 'sheet convert');
    if (own) g._submit(enc);
    return rho;
  }

  /**
   * Exact (clipping-based) deposit of the same tetrahedra: cell-averaged rho/rhobar on the n^3 grid, P0 (weights null) or P1
   * (`weights` = n^3 vertex densities).  Persistent buffer 'sheetx-rho' of g.  Tetrahedra spanning more than EXACT3D_MAXB cells per axis
   * are skipped (their mass is lost; see stats()).  `opts.chunks` splits the dispatch into several submits (GPU watchdog safety).
   */
  densityExact(disp, weights = null, encoder = null, opts = {}) {
    if (weights && typeof weights.beginComputePass === 'function') { encoder = weights; weights = null; }
    const g = this.g, N = this.size, n = this.n;
    const ne = opts.ne || n, maxb = opts.maxb || EXACT3D_MAXB;
    if (ne !== n) throw new Error('densityExact: the Eulerian grid must equal the Lagrangian grid (ne = n) on the lab path');
    const own = !encoder;
    const acc = g.buf('sheetx-acc', g.fbytes), rho = g.buf('sheetx-rho', g.fbytes);
    if (!this.xstats) this.xstats = this.device.createBuffer({ label: 'exact3d stats', size: 16, usage: STORAGE_RW });
    const chunks = own ? Math.max(1, opts.chunks || 1) : 1;
    const per = Math.ceil(N / chunks / 32) * 32;
    let enc = encoder || this.device.createCommandEncoder({ label: 'sheet exact' });
    enc.clearBuffer(acc);
    enc.clearBuffer(this.xstats);
    for (let c = 0; c < chunks; c++) {
      const base = c * per, cnt = Math.min(N, base + per) - base;
      if (cnt <= 0) break;
      const slot = this.ring.write([['u', N], ['u', n], ['u', ne], ['u', CIC_SCALE], ['u', weights ? 1 : 0], ['u', maxb], ['u', base], ['u', 0], ne, ne / n]);
      const bg = makeBindGroup(this.device, this.pipeX.bgl, [this.ring.resource(48), disp, acc, weights || this.dummy, this.xstats], 'exact bg');
      const pass = enc.beginComputePass({ label: 'exact tets' });
      pass.setPipeline(this.pipeX.pipeline);
      pass.setBindGroup(0, bg, [slot]);
      dispatch1d(pass, cnt, 32);
      pass.end();
      if (own && c + 1 < chunks) { g._submit(enc); enc = this.device.createCommandEncoder({ label: 'sheet exact' }); }
    }
    const words = [['u', N], ['u', n], ['u', CIC_SCALE], 0, n];
    g._dispatch(enc, 'sheetx', g.P.cicConvert, g.bg('sheetxconv', g.P.cicConvert, [g.ring.resource(32), acc, rho]), words, N, 'sheet exact convert');
    if (own) g._submit(enc);
    return rho;
  }
  /** Counters of the last densityExact: tetrahedra over the bounding-box clamp, over the clipper's vertex cap, degenerate (skipped). */
  async stats() {
    const s = await readU32(this.device, this.xstats, 4);
    return { skippedBox: s[0], skippedCap: s[1], skippedDegenerate: s[2] };
  }
}
