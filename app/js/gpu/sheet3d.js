// sheet3d.js - phase-space-sheet density of the 3-D lab on the GPU (same quantity as sheet.rs::sheet_density_3d, ss = 1).
//
//   const sheet = new GpuSheet3D(g); await sheet.init();
//   rho = sheet.density(dispBuffer [, encoder])      GPUBuffer n^3 f32, rho/rhobar; dispBuffer = Psi = x - q (box units, interleaved xyz,
//                                                    e.g. g.displacement(gvals, order)); persistent buffer 'sheet-rho' of g
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

import { ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR } from './util.js';
import { CIC_SCALE } from './cosmo3d.js';

const WGSL = /* wgsl */`
${WGSL_LINEAR}
struct SU { count: u32, n: u32, scale: u32, p0: u32, nf: f32 };
@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;

var<private> TETS: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
  vec4<u32>(0u, 1u, 3u, 7u), vec4<u32>(0u, 1u, 5u, 7u), vec4<u32>(0u, 2u, 3u, 7u),
  vec4<u32>(0u, 2u, 6u, 7u), vec4<u32>(0u, 4u, 5u, 7u), vec4<u32>(0u, 4u, 6u, 7u));

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
  var vid: array<u32, 8>;          // global (wrapped) vertex id
  for (var b = 0u; b < 8u; b = b + 1u) {
    let bx = b & 1u; let by = (b >> 1u) & 1u; let bz = (b >> 2u) & 1u;
    let idx = ((u32(ix + i32(bx)) & mask) * n + (u32(iy + i32(by)) & mask)) * n + (u32(iz + i32(bz)) & mask);
    vid[b] = idx;
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
      l4[k] = lc[b]; p4[k] = pw[b]; id4[k] = vid[b];
      bits[k] = vec3<i32>(i32(b & 1u), i32((b >> 1u) & 1u), i32((b >> 2u) & 1u));
    }
    let vol6 = dot(l4[1] - l4[0], cross(l4[2] - l4[0], l4[3] - l4[0]));
    if (abs(vol6) < 1e-9) { continue; }
    let dens = 1.0 / abs(vol6);
    let q = u32(min(dens * f32(su.scale) + 0.5, 2147483648.0));
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
            atomicAdd(&acc[idx], q);
          }
        }
      }
    }
  }
}
`;

const pipeCache = new WeakMap();
function getPipeline(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const m = await compile(device, WGSL, 'sheet3d');
      return makePipeline(device, m, 'sheet_tet', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'sheet_tet');
    })();
    pipeCache.set(device, p);
  }
  return p;
}

export class GpuSheet3D {
  constructor(g) { this.g = g; this.device = g.device; this.n = g.n; this.size = g.size; this.ring = ParamRing.get(g.device); }
  async init() { this.pipe = await getPipeline(this.device); return this; }

  density(disp, encoder = null) {
    const g = this.g, N = this.size, n = this.n;
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'sheet' });
    const acc = g.buf('sheet-acc', g.fbytes), rho = g.buf('sheet-rho', g.fbytes);
    enc.clearBuffer(acc);
    const slot = this.ring.write([['u', N], ['u', n], ['u', CIC_SCALE], 0, n]);
    const bg = makeBindGroup(this.device, this.pipe.bgl, [this.ring.resource(32), disp, acc], 'sheet bg');
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
}
