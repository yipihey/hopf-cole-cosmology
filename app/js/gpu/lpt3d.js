// lpt3d.js - GPU-resident nLPT for the 3-D lab: the term build, shell-crossing search, the Helmholtz split and the
// transverse correction of the Legendre (Hopf-Cole) inversion.  Everything runs in WGSL compute shaders on the
// GpuCosmo3D instance `g` (shared device, FFT, scratch buffers); the only CPU inputs are delta0 (once per IC) and the
// growth factors g_tau(D) (a few numbers per D).
//
//   const lpt = new GpuLpt3D(g); await lpt.init();
//   await lpt.build(delta0, spec, order)             spec = getUnmergedSpec(sim) (terms.js); builds S^tau, tau = 1..13 (4LPT),
//                                                    installs them in g (g.terms/g.orders, so g.positions/displacement/cicDensity work),
//                                                    writes the Zel'dovich potential into g's 'phi0' (statistics registered)
//   {frac} = await lpt.legendre(gvals, order, D)     Helmholtz split of Psi(D) = sum g_tau S^tau: 'phi-eff' (n^3) and 'psiT' (3 n^3) buffers
//                                                    of g, window statistics registered, frac = rms|Psi_T|/rms|Psi_L|
//   lpt.transverse(D, enc)                           (after g.hopfCole(nu, D, enc, 'phi-eff')) replaces g's 'hc-delta' by the
//                                                    transverse-corrected density det(dq/dx) - 1, q = q_L - Psi_T(q_L)
//   Dsc = await lpt.shellCrossing(order, gfun, dmax) first shell crossing (gfun(D) -> g_tau(D) per term; dmax = largest D)
//   Dsc1 = await lpt.zeldovichDsc()                  1LPT estimate 1/max(-lambda_min(grad S^1))
//   du = await lpt.velocityJump()                    max|S^1| - min|S^1|
//   lpt.destroy()
//
// Algorithm (term tau with parents from the raw recursion of growth.rs::term_specs):
//   source  mu2(A,B) = 1/2 [trA trB - tr(AB)],  mu3(A,B,C) = 1/6 [...],  T_i = eps_ijk A_lj B_lk   (pointwise from the stored
//           gradient tensors M_ab = d_b S_a of the parents), FFT,
//   solve   S_a = -i k_a s^/k^2 (longitudinal)  or  S^ = i (k x T^)/k^2 (transverse), M_ab = i k_b S_a,
//           with the derivative-safe wavenumber kd (Nyquist component zeroed) in the odd operators and the full k^2, k = 0 mode zeroed
//           (grid.rs conventions),
//   inverse FFT of two real fields per transform (z = A^ + i B^ of two Hermitian spectra gives A in Re, B in Im), so a term costs
//           1 forward FFT (3 for the transverse terms) and 2 (no tensor needed) to 6 inverse FFTs.
//
// Memory (f32; N = n^3 cells):  a term field is 12 N bytes (24 MiB at n = 128), 13 of them for 4LPT (312 MiB); a gradient tensor set is
//   36 N bytes (72 MiB at 128^3) and is kept only for terms that are parents of a later term, until their last child is built
//   (4LPT: terms 1-5, 360 MiB at the peak); complex work buffers 6 x 8 N (96 MiB, kept for the Helmholtz split), FFT scratch 8 N.
//   Peak of a 4LPT build at 128^3: 312 + 360 + 96 + 16 + the GpuCosmo3D working buffers (~190 MiB) = about 1.0 GiB (1.1 GiB when an order increase rebuilds while the old terms still exist); steady state
//   after the build: 688 MiB (measured: 312 terms + 96 complex + 280 scratch/field buffers); n = 64: 1/8 of these.
//   Per-binding limit: a tensor set is 72 MiB at n = 128 (< the 128 MiB default maxStorageBufferBindingSize); n = 256 would need
//   576 MiB per binding and is not supported (the constructor of GpuCosmo3D already refuses it).

import { ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';
import { OP, RED_PRELUDE, stage1Layout } from './reduce.js';

const NONE = 0xFFFFFFFF;

// --------------------------------------------------------------------------------------------------------------------
const BUILD_WGSL = /* wgsl */`
${WGSL_LINEAR}
const TWO_PI: f32 = 6.283185307179586;
fn mfreq(i: u32, n: u32) -> f32 { if (i < n / 2u) { return f32(i); } return f32(i) - f32(n); }
fn kfull(i: u32, n: u32) -> f32 { return TWO_PI * mfreq(i, n); }
fn kder(i: u32, n: u32) -> f32 { if (i == n / 2u) { return 0.0; } return TWO_PI * mfreq(i, n); }
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cI(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(-a.y, a.x); }

// ---- spectral solve + pack two real fields into one complex spectrum
struct SP { n: u32, mode: u32, j1: u32, j2: u32, sgn: f32 };
@group(0) @binding(0) var<uniform> sp: SP;
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> h1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> h2: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> zo: array<vec2<f32>>;

// field j of the term spectrum: 0..2 = S_a, 3..11 = M_ab = i k_b S_a (j = 3 + 3a + b), 12 = phi (Zel'dovich potential), NONE = 0
fn fld(j: u32, kd: vec3<f32>, S: array<vec2<f32>, 3>, phi: vec2<f32>) -> vec2<f32> {
  if (j == 0xFFFFFFFFu) { return vec2<f32>(0.0); }
  var Sl = S;
  if (j < 3u) { return Sl[j]; }
  if (j == 12u) { return phi; }
  let c = j - 3u;
  let a = c / 3u; let b = c % 3u;
  var kb = kd.x;
  if (b == 1u) { kb = kd.y; } else if (b == 2u) { kb = kd.z; }
  return cI(Sl[a]) * kb;
}

@compute @workgroup_size(256)
fn spec_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  let n = sp.n;
  if (i >= n * n * n) { return; }
  let ix = i / (n * n); let iy = (i / n) % n; let iz = i % n;
  let kv = vec3<f32>(kfull(ix, n), kfull(iy, n), kfull(iz, n));
  let kd = vec3<f32>(kder(ix, n), kder(iy, n), kder(iz, n));
  let k2 = dot(kv, kv);
  if (k2 == 0.0) { zo[i] = vec2<f32>(0.0); return; }
  var S: array<vec2<f32>, 3>;
  var phi = vec2<f32>(0.0);
  if (sp.mode == 0u) {
    let s = h0[i] * sp.sgn;
    S[0] = cmul(vec2<f32>(0.0, -kd.x / k2), s);
    S[1] = cmul(vec2<f32>(0.0, -kd.y / k2), s);
    S[2] = cmul(vec2<f32>(0.0, -kd.z / k2), s);
    phi = s / k2;
  } else {
    let t0 = h0[i]; let t1 = h1[i]; let t2 = h2[i];
    S[0] = cI(kd.y * t2 - kd.z * t1) / k2;
    S[1] = cI(kd.z * t0 - kd.x * t2) / k2;
    S[2] = cI(kd.x * t1 - kd.y * t0) / k2;
  }
  zo[i] = fld(sp.j1, kd, S, phi) + cI(fld(sp.j2, kd, S, phi));
}

// ---- scatter the two real fields of an inverse transform into the term / tensor / potential buffers
struct SU { count: u32, j1: u32, j2: u32, sym: u32 };
@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var<storage, read> zi: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> sdst: array<f32>;
@group(0) @binding(3) var<storage, read_write> tdst: array<f32>;
@group(0) @binding(4) var<storage, read_write> pdst: array<f32>;
fn put(j: u32, i: u32, v: f32) {
  if (j == 0xFFFFFFFFu) { return; }
  if (j < 3u) { sdst[3u * i + j] = v; return; }
  if (j == 12u) { pdst[i] = v; return; }
  let c = j - 3u;
  tdst[9u * i + c] = v;
  let a = c / 3u; let b = c % 3u;
  if (su.sym == 1u && a != b) { tdst[9u * i + b * 3u + a] = v; }
}
@compute @workgroup_size(256)
fn spec_unpack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= su.count) { return; }
  let z = zi[i];
  put(su.j1, i, z.x);
  put(su.j2, i, z.y);
}

// ---- pointwise sources from the parents' gradient tensors (M[a*3+b] = d_b S_a)
struct SrcU { count: u32, comp: u32 };
@group(0) @binding(0) var<uniform> srcu: SrcU;
@group(0) @binding(1) var<storage, read> ta: array<f32>;
@group(0) @binding(2) var<storage, read> tb: array<f32>;
@group(0) @binding(3) var<storage, read> tc: array<f32>;
@group(0) @binding(4) var<storage, read_write> so: array<vec2<f32>>;

fn ld9(buf: u32, i: u32) -> array<f32, 9> {
  var m: array<f32, 9>;
  for (var c = 0u; c < 9u; c = c + 1u) {
    if (buf == 0u) { m[c] = ta[9u * i + c]; } else if (buf == 1u) { m[c] = tb[9u * i + c]; } else { m[c] = tc[9u * i + c]; }
  }
  return m;
}
fn trace(A: array<f32, 9>) -> f32 { return A[0] + A[4] + A[8]; }
fn tr2(A: array<f32, 9>, B: array<f32, 9>) -> f32 {
  var a = A; var b = B; var s = 0.0;
  for (var i = 0u; i < 3u; i = i + 1u) { for (var j = 0u; j < 3u; j = j + 1u) { s = s + a[i * 3u + j] * b[j * 3u + i]; } }
  return s;
}
fn tr3(A: array<f32, 9>, B: array<f32, 9>, C: array<f32, 9>) -> f32 {
  var a = A; var b = B; var c = C; var s = 0.0;
  for (var i = 0u; i < 3u; i = i + 1u) { for (var j = 0u; j < 3u; j = j + 1u) { for (var k = 0u; k < 3u; k = k + 1u) {
    s = s + a[i * 3u + j] * b[j * 3u + k] * c[k * 3u + i];
  } } }
  return s;
}
@compute @workgroup_size(256)
fn src_mu2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= srcu.count) { return; }
  let A = ld9(0u, i); let B = ld9(1u, i);
  so[i] = vec2<f32>(0.5 * (trace(A) * trace(B) - tr2(A, B)), 0.0);
}
@compute @workgroup_size(256)
fn src_mu3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= srcu.count) { return; }
  let A = ld9(0u, i); let B = ld9(1u, i); let C = ld9(2u, i);
  let ta_ = trace(A); let tb_ = trace(B); let tc_ = trace(C);
  let v = (ta_ * tb_ * tc_ - ta_ * tr2(B, C) - tb_ * tr2(A, C) - tc_ * tr2(A, B) + tr3(A, B, C) + tr3(A, C, B)) / 6.0;
  so[i] = vec2<f32>(v, 0.0);
}
@compute @workgroup_size(256)
fn src_curl(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= srcu.count) { return; }
  let A = ld9(0u, i); let B = ld9(1u, i);
  var a = A; var b = B;
  let c = srcu.comp;
  let j = (c + 1u) % 3u; let k = (c + 2u) % 3u;
  var v = 0.0;
  for (var l = 0u; l < 3u; l = l + 1u) { v = v + a[l * 3u + j] * b[l * 3u + k] - a[l * 3u + k] * b[l * 3u + j]; }
  so[i] = vec2<f32>(v, 0.0);
}

// ---- real component -> complex (for the forward transforms of the displacement components)
struct PcU { count: u32, comp: u32 };
@group(0) @binding(0) var<uniform> pcu: PcU;
@group(0) @binding(1) var<storage, read> pci: array<f32>;
@group(0) @binding(2) var<storage, read_write> pco: array<vec2<f32>>;
@compute @workgroup_size(256)
fn pack_comp(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= pcu.count) { return; }
  pco[i] = vec2<f32>(pci[3u * i + pcu.comp], 0.0);
}

// ---- Helmholtz split in Fourier space: z1 = phi_eff^ + i PsiL_x^,  z2 = PsiL_y^ + i PsiL_z^
struct SpU { n: u32, part: u32, p0: u32, p1: u32, D: f32 };
@group(0) @binding(0) var<uniform> spu: SpU;
@group(0) @binding(1) var<storage, read> g0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> g1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> g2: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> go: array<vec2<f32>>;
@compute @workgroup_size(256)
fn split_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  let n = spu.n;
  if (i >= n * n * n) { return; }
  let ix = i / (n * n); let iy = (i / n) % n; let iz = i % n;
  let kv = vec3<f32>(kfull(ix, n), kfull(iy, n), kfull(iz, n));
  let kd = vec3<f32>(kder(ix, n), kder(iy, n), kder(iz, n));
  let k2 = dot(kv, kv);
  if (k2 == 0.0) { go[i] = vec2<f32>(0.0); return; }
  let div = cI(g0[i] * kd.x + g1[i] * kd.y + g2[i] * kd.z);                // (div Psi)^ = i k.Psi^
  var dd = spu.D;
  if (abs(dd) < 1e-12) { dd = 1.0; }
  let il = div / k2;                                                       // PsiL_a^ = -i k_a div^/k^2
  let pl = vec3<f32>(0.0);
  var a = vec2<f32>(0.0); var b = vec2<f32>(0.0);
  if (spu.part == 0u) {
    a = div / (k2 * dd);                                                   // phi_eff^ = div^/(D k^2)
    b = cmul(vec2<f32>(0.0, -kd.x), il);
  } else {
    a = cmul(vec2<f32>(0.0, -kd.y), il);
    b = cmul(vec2<f32>(0.0, -kd.z), il);
  }
  go[i] = a + cI(b);
}

struct SuU { count: u32, part: u32 };
@group(0) @binding(0) var<uniform> suu: SuU;
@group(0) @binding(1) var<storage, read> wi: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> dspi: array<f32>;
@group(0) @binding(3) var<storage, read_write> phio: array<f32>;
@group(0) @binding(4) var<storage, read_write> psito: array<f32>;
@compute @workgroup_size(256)
fn split_unpack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= suu.count) { return; }
  let w = wi[i];
  if (suu.part == 0u) {
    phio[i] = w.x;
    psito[3u * i] = dspi[3u * i] - w.y;
  } else {
    psito[3u * i + 1u] = dspi[3u * i + 1u] - w.x;
    psito[3u * i + 2u] = dspi[3u * i + 2u] - w.y;
  }
}
`;

// --------------------------------------------------------------------------------------------------------------------
const AUX_WGSL = /* wgsl */`
${RED_PRELUDE}
@group(0) @binding(2) var<storage, read> fld3: array<f32>;
@group(0) @binding(3) var<storage, read> fld3b: array<f32>;
@group(0) @binding(2) var<storage, read> fsc: array<f32>;
@group(0) @binding(4) var<storage, read_write> fout3: array<f32>;
@group(0) @binding(3) var<storage, read_write> fout1: array<f32>;

fn wrapi(x: i32, n: u32) -> u32 { return u32(x & (i32(n) - 1)); }
fn p3(x: i32, y: i32, z: i32, n: u32) -> vec3<f32> {
  let b = 3u * ((wrapi(x, n) * n + wrapi(y, n)) * n + wrapi(z, n));
  return vec3<f32>(fld3[b], fld3[b + 1u], fld3[b + 2u]);
}
// columns of the gradient tensor, col_b = d_b Psi (4th-order central differences; box units)
fn grad_cols(x: i32, y: i32, z: i32, n: u32) -> mat3x3<f32> {
  let s = f32(n) / 12.0;
  let cx = (8.0 * (p3(x + 1, y, z, n) - p3(x - 1, y, z, n)) - (p3(x + 2, y, z, n) - p3(x - 2, y, z, n))) * s;
  let cy = (8.0 * (p3(x, y + 1, z, n) - p3(x, y - 1, z, n)) - (p3(x, y + 2, z, n) - p3(x, y - 2, z, n))) * s;
  let cz = (8.0 * (p3(x, y, z + 1, n) - p3(x, y, z - 1, n)) - (p3(x, y, z + 2, n) - p3(x, y, z - 2, n))) * s;
  return mat3x3<f32>(cx, cy, cz);
}
fn cell_xyz(i: u32, n: u32) -> vec3<i32> { return vec3<i32>(i32(i / (n * n)), i32((i / n) % n), i32(i % n)); }

// min_q det(I + grad Psi)
@compute @workgroup_size(256)
fn st_jac(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  var v = ident(ru.ops);
  if (i < ru.count) {
    let c = cell_xyz(i, ru.n);
    let G = grad_cols(c.x, c.y, c.z, ru.n);
    let m = mat3x3<f32>(G[0] + vec3<f32>(1.0, 0.0, 0.0), G[1] + vec3<f32>(0.0, 1.0, 0.0), G[2] + vec3<f32>(0.0, 0.0, 1.0));
    v = vec4<f32>(determinant(m), 0.0, 0.0, 0.0);
  }
  red_store(v, lid, wid, nwg);
}

// max over cells of -lambda_min(sym(grad S1)): 1/(Zel'dovich shell-crossing time)
@compute @workgroup_size(256)
fn st_zel(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  var v = ident(ru.ops);
  if (i < ru.count) {
    let c = cell_xyz(i, ru.n);
    let G = grad_cols(c.x, c.y, c.z, ru.n);       // G[b][a] = M_ab
    let a0 = G[0][0]; let a1 = G[1][1]; let a2 = G[2][2];
    let a3 = 0.5 * (G[1][0] + G[0][1]); let a4 = 0.5 * (G[2][0] + G[0][2]); let a5 = 0.5 * (G[2][1] + G[1][2]);
    let q = (a0 + a1 + a2) / 3.0;
    let p2 = (a0 - q) * (a0 - q) + (a1 - q) * (a1 - q) + (a2 - q) * (a2 - q) + 2.0 * (a3 * a3 + a4 * a4 + a5 * a5);
    let pp = sqrt(p2 / 6.0);
    var lmin = q;
    if (pp > 1e-30) {
      let b0 = (a0 - q) / pp; let b1 = (a1 - q) / pp; let b2 = (a2 - q) / pp;
      let b3 = a3 / pp; let b4 = a4 / pp; let b5 = a5 / pp;
      let detb = b0 * (b1 * b2 - b5 * b5) - b3 * (b3 * b2 - b5 * b4) + b4 * (b3 * b5 - b1 * b4);
      let r = clamp(detb / 2.0, -1.0, 1.0);
      let ph = acos(r) / 3.0;
      lmin = q + 2.0 * pp * cos(ph + 2.0943951023931953);
    }
    v = vec4<f32>(-lmin, 0.0, 0.0, 0.0);
  }
  red_store(v, lid, wid, nwg);
}

// min / max |S1|
@compute @workgroup_size(256)
fn st_vel(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  var v = ident(ru.ops);
  if (i < ru.count) {
    let s = vec3<f32>(fld3[3u * i], fld3[3u * i + 1u], fld3[3u * i + 2u]);
    let m = length(s);
    v = vec4<f32>(m, m, 0.0, 0.0);
  }
  red_store(v, lid, wid, nwg);
}

// sum |Psi_T|^2, sum |Psi - Psi_T|^2  (fld3 = Psi, fld3b = Psi_T)
@compute @workgroup_size(256)
fn st_split(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  var v = ident(ru.ops);
  if (i < ru.count) {
    let p = vec3<f32>(fld3[3u * i], fld3[3u * i + 1u], fld3[3u * i + 2u]);
    let t = vec3<f32>(fld3b[3u * i], fld3b[3u * i + 1u], fld3b[3u * i + 2u]);
    let l = p - t;
    v = vec4<f32>(dot(t, t), dot(l, l), 0.0, 0.0);
  }
  red_store(v, lid, wid, nwg);
}

// ---- transverse correction: dq = q - x with q = q_L - Psi_T(q_L), q_L = x - D u(x), u = grad Phi_v (fsc = Phi_v, fld3b = Psi_T)
fn fsca(x: i32, y: i32, z: i32, n: u32) -> f32 { return fsc[(wrapi(x, n) * n + wrapi(y, n)) * n + wrapi(z, n)]; }
@compute @workgroup_size(256)
fn tc_disp(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= ru.count) { return; }
  let n = ru.n;
  let c = cell_xyz(i, n);
  let nf = f32(n);
  let s = nf / 12.0;
  let u = vec3<f32>(
    (8.0 * (fsca(c.x + 1, c.y, c.z, n) - fsca(c.x - 1, c.y, c.z, n)) - (fsca(c.x + 2, c.y, c.z, n) - fsca(c.x - 2, c.y, c.z, n))) * s,
    (8.0 * (fsca(c.x, c.y + 1, c.z, n) - fsca(c.x, c.y - 1, c.z, n)) - (fsca(c.x, c.y + 2, c.z, n) - fsca(c.x, c.y - 2, c.z, n))) * s,
    (8.0 * (fsca(c.x, c.y, c.z + 1, n) - fsca(c.x, c.y, c.z - 1, n)) - (fsca(c.x, c.y, c.z + 2, n) - fsca(c.x, c.y, c.z - 2, n))) * s);
  let D = ru.f.x;
  let ql = vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) - D * u * nf;          // q_L in cell units
  let fl = floor(ql);
  let w1 = ql - fl;
  let i0 = vec3<i32>(fl);
  var t = vec3<f32>(0.0);
  for (var k = 0u; k < 8u; k = k + 1u) {
    let bx = k & 1u; let by = (k >> 1u) & 1u; let bz = (k >> 2u) & 1u;
    let w = select(1.0 - w1.x, w1.x, bx == 1u) * select(1.0 - w1.y, w1.y, by == 1u) * select(1.0 - w1.z, w1.z, bz == 1u);
    let j = 3u * ((wrapi(i0.x + i32(bx), n) * n + wrapi(i0.y + i32(by), n)) * n + wrapi(i0.z + i32(bz), n));
    t = t + w * vec3<f32>(fld3b[j], fld3b[j + 1u], fld3b[j + 2u]);
  }
  let dq = -D * u - t;
  fout3[3u * i] = dq.x; fout3[3u * i + 1u] = dq.y; fout3[3u * i + 2u] = dq.z;
}
// delta = det(I + grad dq) - 1  (fld3 = dq)
@compute @workgroup_size(256)
fn tc_delta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= ru.count) { return; }
  let c = cell_xyz(i, ru.n);
  let G = grad_cols(c.x, c.y, c.z, ru.n);
  let m = mat3x3<f32>(G[0] + vec3<f32>(1.0, 0.0, 0.0), G[1] + vec3<f32>(0.0, 1.0, 0.0), G[2] + vec3<f32>(0.0, 0.0, 1.0));
  fout1[i] = determinant(m) - 1.0;
}
`;

const pipeCache = new WeakMap();
function getPipelines(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const [mb, ma] = await Promise.all([compile(device, BUILD_WGSL, 'lpt3d build'), compile(device, AUX_WGSL, 'lpt3d aux')]);
      const U = U_ENTRY, R = RO_ENTRY, W = RW_ENTRY;
      const mk = (m, name, entries) => makePipeline(device, m, name, entries, name);
      const src = [U(0), R(1), R(2), R(3), W(4)];
      return {
        specPack: mk(mb, 'spec_pack', [U(0), R(1), R(2), R(3), W(4)]),
        specUnpack: mk(mb, 'spec_unpack', [U(0), R(1), W(2), W(3), W(4)]),
        mu2: mk(mb, 'src_mu2', src), mu3: mk(mb, 'src_mu3', src), curl: mk(mb, 'src_curl', src),
        packComp: mk(mb, 'pack_comp', [U(0), R(1), W(2)]),
        splitPack: mk(mb, 'split_pack', [U(0), R(1), R(2), R(3), W(4)]),
        splitUnpack: mk(mb, 'split_unpack', [U(0), R(1), R(2), W(3), W(4)]),
        stJac: mk(ma, 'st_jac', stage1Layout(1)),
        stZel: mk(ma, 'st_zel', stage1Layout(1)),
        stVel: mk(ma, 'st_vel', stage1Layout(1)),
        stSplit: mk(ma, 'st_split', stage1Layout(2)),
        tcDisp: mk(ma, 'tc_disp', [U(0), R(2), R(3), W(4)]),
        tcDelta: mk(ma, 'tc_delta', [U(0), R(2), W(3)]),
      };
    })();
    pipeCache.set(device, p);
  }
  return p;
}

export class GpuLpt3D {
  constructor(g) {
    this.g = g; this.device = g.device; this.n = g.n; this.size = g.size;
    this.ring = ParamRing.get(g.device);
    this.tensors = new Map();
    this.cbuf = {};                 // complex work buffers
    this.built = null;              // {order, count}
    this.lastTimings = {};
  }
  async init() { this.P = await getPipelines(this.device); return this; }

  // ------------------------------------------------------------------ helpers
  _cplx(name) {
    if (!this.cbuf[name]) this.cbuf[name] = this.device.createBuffer({ label: 'lpt ' + name, size: 8 * this.size, usage: STORAGE_RW });
    return this.cbuf[name];
  }
  /** Like _pass with explicit binding numbers: bindings = [[binding, resource], ...] (the uniform is binding 0). */
  _passB(enc, pipe, bindings, words, count, label, resSize = 48) {
    const slot = this.ring.write(words);
    const bg = this.device.createBindGroup({ label, layout: pipe.bgl, entries: [{ binding: 0, resource: this.ring.resource(resSize) }, ...bindings.map(([binding, r]) => ({ binding, resource: { buffer: r } }))] });
    const pass = enc.beginComputePass({ label });
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bg, [slot]);
    dispatch1d(pass, count, 256);
    pass.end();
  }
  _pass(enc, pipe, resources, words, count, label, resSize = 32) {
    const slot = this.ring.write(words);
    const bg = makeBindGroup(this.device, pipe.bgl, [this.ring.resource(resSize), ...resources], label);
    const pass = enc.beginComputePass({ label });
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bg, [slot]);
    dispatch1d(pass, count, 256);
    pass.end();
  }

  // ------------------------------------------------------------------ build
  /**
   * Build S^tau for tau in spec with order <= `order` from delta0 (Float32Array n^3, the linear density at D = 1).
   * Installs the term buffers in g and the Zel'dovich potential in g's 'phi0'.  Returns timing info.
   */
  async build(delta0, spec, order) {
    const g = this.g, d = this.device, n = this.n, N = this.size;
    const t0 = performance.now();
    let nT = 0;
    while (nT < spec.count && spec.orders[nT] <= order) nT++;
    if (nT === 0) throw new Error('GpuLpt3D.build: no terms');
    const parents = (t) => (spec.kind[t] === 1 || spec.kind[t] === 3 ? [spec.a[t], spec.b[t]] : spec.kind[t] === 2 ? [spec.a[t], spec.b[t], spec.c[t]] : []);
    const lastUse = new Map();
    for (let t = 0; t < nT; t++) for (const p of parents(t)) lastUse.set(p, t);
    for (const b of this.tensors.values()) b.destroy();
    this.tensors.clear();
    d.pushErrorScope('out-of-memory');
    d.pushErrorScope('validation');
    const termBufs = [];
    let err = null;
    try {
      for (let t = 0; t < nT; t++) termBufs.push(d.createBuffer({ label: 'term' + t, size: 12 * N, usage: STORAGE_RW }));
      const cA = this._cplx('cA'), h0 = this._cplx('h0'), h1 = this._cplx('h1'), h2 = this._cplx('h2'), zc = this._cplx('z'), wc = this._cplx('w');
      const hs = [h0, h1, h2];
      const phi0 = g.buf('phi0', 4 * N);
      const dummy = d.createBuffer({ label: 'lpt dummy', size: 16, usage: STORAGE_RW });
      // delta0^ -> h0
      {
        const enc = d.createCommandEncoder({ label: 'lpt delta0' });
        const b = g.uploadField(delta0);
        g.fft.packReal(b, cA, enc, 0);
        g.fft.forward(cA, h0, enc);
        d.queue.submit([enc.finish()]);
      }
      for (let t = 0; t < nT; t++) {
        const enc = d.createCommandEncoder({ label: 'lpt term ' + t });
        const kind = spec.kind[t];
        const needT = lastUse.has(t);
        let tens = null;
        if (needT) { tens = d.createBuffer({ label: 'tensor' + t, size: 36 * N, usage: STORAGE_RW }); this.tensors.set(t, tens); }
        let mode = 0, sgn = 1;
        if (kind === 0) { mode = 0; sgn = -1; }                              // s = div S^1 = -delta0
        else if (kind === 1 || kind === 2) {
          const A = this.tensors.get(spec.a[t]), B = this.tensors.get(spec.b[t]), C = kind === 2 ? this.tensors.get(spec.c[t]) : A;
          this._pass(enc, kind === 1 ? this.P.mu2 : this.P.mu3, [A, B, C, cA], [['u', N], 0], N, 'source t' + t, 16);
          g.fft.forward(cA, h0, enc);
        } else {
          const A = this.tensors.get(spec.a[t]), B = this.tensors.get(spec.b[t]);
          for (let c = 0; c < 3; c++) {
            this._pass(enc, this.P.curl, [A, B, B, cA], [['u', N], ['u', c]], N, 'curl t' + t, 16);
            g.fft.forward(cA, hs[c], enc);
          }
          mode = 1;
        }
        // real fields to produce: S (0,1,2), tensor (symmetric for gradients of a potential), phi for the linear term
        const sym = kind !== 3;
        const fields = [0, 1, 2];
        if (needT) fields.push(...(sym ? [3, 4, 5, 7, 8, 11] : [3, 4, 5, 6, 7, 8, 9, 10, 11]));
        if (t === 0) fields.push(12);
        for (let q = 0; q < fields.length; q += 2) {
          const j1 = fields[q], j2 = q + 1 < fields.length ? fields[q + 1] : NONE;
          this._pass(enc, this.P.specPack, [h0, h1, h2, zc], [['u', n], ['u', mode], ['u', j1], ['u', j2], sgn], N, 'solve t' + t);
          g.fft.inverse(zc, wc, enc);
          this._pass(enc, this.P.specUnpack, [wc, termBufs[t], tens || dummy, phi0], [['u', N], ['u', j1], ['u', j2], ['u', sym ? 1 : 0]], N, 'scatter t' + t, 16);
        }
        d.queue.submit([enc.finish()]);
        // free tensors whose last child is built
        for (const [p, last] of lastUse) if (last === t && this.tensors.has(p)) { this.tensors.get(p).destroy(); this.tensors.delete(p); }
      }
      dummy.destroy();
    } catch (e) { err = e; }
    const ve = await d.popErrorScope();
    const oe = await d.popErrorScope();
    if (err || ve || oe) {
      for (const b of termBufs) b.destroy();
      for (const b of this.tensors.values()) b.destroy();
      this.tensors.clear();
      throw new Error('GPU nLPT build failed: ' + (err ? err.message : oe ? oe.message : ve.message));
    }
    for (const b of this.tensors.values()) b.destroy();         // (none left; defensive)
    this.tensors.clear();
    await d.queue.onSubmittedWorkDone();
    g.installTerms(termBufs, spec.orders.slice(0, nT));
    await g.potentialStats('phi0');
    this.built = { order, count: nT };
    this.lastTimings.build = performance.now() - t0;
    return { count: nT, ms: this.lastTimings.build };
  }

  // ------------------------------------------------------------------ Helmholtz split / Legendre source
  /**
   * Psi(D) = sum g_tau S^tau (orders <= `order`), Psi_L = -D grad phi_eff its longitudinal part.
   * Fills g's 'phi-eff' (n^3) and 'psiT' (= Psi - Psi_L, 3 n^3); registers the window statistics of phi_eff.
   * Returns {frac} = rms|Psi_T| / rms|Psi_L|.
   */
  async legendre(gvals, order, D) {
    const g = this.g, d = this.device, n = this.n, N = this.size;
    const enc = d.createCommandEncoder({ label: 'legendre split' });
    const disp = g.displacement(gvals, order, enc, 'disp');
    const cA = this._cplx('cA'), hs = [this._cplx('h0'), this._cplx('h1'), this._cplx('h2')], zc = this._cplx('z'), wc = this._cplx('w');
    const phiEff = g.buf('phi-eff', 4 * N), psiT = g.buf('psiT', 12 * N);
    for (let c = 0; c < 3; c++) {
      this._pass(enc, this.P.packComp, [disp, cA], [['u', N], ['u', c]], N, 'pack comp', 16);
      g.fft.forward(cA, hs[c], enc);
    }
    for (let part = 0; part < 2; part++) {
      this._pass(enc, this.P.splitPack, [hs[0], hs[1], hs[2], zc], [['u', n], ['u', part], 0, 0, D], N, 'split', 32);
      g.fft.inverse(zc, wc, enc);
      this._pass(enc, this.P.splitUnpack, [wc, disp, phiEff, psiT], [['u', N], ['u', part]], N, 'split scatter', 16);
    }
    g.red.run(enc, g.P.stPhi, [phiEff], N, 0, [OP.MIN, OP.MAX, OP.MAX, OP.SUM], { n }, 'phi-eff stats');
    g.red.run(enc, this.P.stSplit, [disp, psiT], N, 1, [OP.SUM, OP.SUM, OP.SUM, OP.SUM], { n }, 'split stats');
    const r = await g.red.read(2, enc);
    g.setPotentialStats('phi-eff', r, 0);
    const frac = r[5] > 0 ? Math.sqrt(r[4] / r[5]) : 0;
    return { frac };
  }

  /**
   * Transverse correction of the Hopf-Cole density: after g.hopfCole(nu, D, enc, 'phi-eff') recorded in `enc`, record
   * q_L = x - D u, q = q_L - Psi_T(q_L), delta = det(dq/dx) - 1 into g's 'hc-delta'.
   */
  transverse(D, enc) {
    const g = this.g, N = this.size, n = this.n;
    const phi = g.buf('hc-phi', 4 * N), delta = g.buf('hc-delta', 4 * N), psiT = g.buf('psiT', 12 * N), dq = g.buf('tc-dq', 12 * N);
    const w = [['u', N], ['u', n], 0, 0, 0, 0, 0, 0, D, 0, 0, 0];
    this._passB(enc, this.P.tcDisp, [[2, phi], [3, psiT], [4, dq]], w, N, 'tc disp');
    this._passB(enc, this.P.tcDelta, [[2, dq], [3, delta]], w, N, 'tc delta');
  }

  // ------------------------------------------------------------------ shell crossing
  /** min_q J(q, D_i) for several D in ONE submit and one readback (g_tau(D) from gfun). */
  async minJacobians(Ds, order, gfun) {
    const g = this.g, N = this.size, n = this.n;
    const enc = this.device.createCommandEncoder({ label: 'min J' });
    const slots = g.red.slots;
    if (Ds.length > slots) throw new Error('minJacobians: too many D values per batch');
    Ds.forEach((D, i) => {
      const disp = g.displacement(gfun(D), order, enc, 'sc-disp');
      g.red.run(enc, this.P.stJac, [disp], N, i, [OP.MIN, OP.SUM, OP.SUM, OP.SUM], { n }, 'min J');
    });
    const r = await g.red.read(Ds.length, enc);
    return Ds.map((_, i) => r[4 * i]);
  }

  /** 1LPT (Zel'dovich) shell-crossing time 1/max_q(-lambda_min(grad S^1)) (Infinity if no collapsing direction). */
  async zeldovichDsc() {
    const g = this.g, N = this.size, n = this.n;
    const enc = this.device.createCommandEncoder({ label: 'zeldovich' });
    g.red.run(enc, this.P.stZel, [g.terms[0]], N, 0, [OP.MAX, OP.SUM, OP.SUM, OP.SUM], { n }, 'zeldovich');
    const r = await g.red.read(1, enc);
    return r[0] > 0 ? 1 / r[0] : Infinity;
  }

  /**
   * First shell-crossing time D_sc of the order-`order` map: the smallest D with min_q J(q, D) <= 0, found as in
   * lpt.rs::shell_crossing (geometric bracket from half the Zel'dovich time, factor 1.6) but with batched evaluations:
   * the bracket scan runs 12 D values per submit, the refinement 4 rounds of 7 interior points (8^4 = 4096-fold reduction).
   */
  async shellCrossing(order, gfun, dmax = Infinity) {
    const zel = await this.zeldovichDsc();
    if (order <= 1) return zel;
    let dd = 0.5 * Math.max(Math.min(zel, 1e3), 1e-3);
    const cands = [];
    for (let i = 0; i < 40; i++) { if (dd > dmax) break; cands.push(dd); dd *= 1.6; }
    let lo = 0, hi = 0;
    for (let s = 0; s < cands.length && hi === 0; s += 12) {
      const chunk = cands.slice(s, s + 12);
      const mj = await this.minJacobians(chunk, order, gfun);
      const k = mj.findIndex((v) => v <= 0);
      if (k >= 0) { hi = chunk[k]; lo = k > 0 ? chunk[k - 1] : (s > 0 ? cands[s - 1] : 0); }
    }
    if (hi === 0) return Infinity;
    for (let round = 0; round < 4; round++) {
      const pts = [];
      for (let j = 1; j <= 7; j++) pts.push(lo + (hi - lo) * j / 8);
      const mj = await this.minJacobians(pts, order, gfun);
      const k = mj.findIndex((v) => v <= 0);
      if (k >= 0) { hi = pts[k]; if (k > 0) lo = pts[k - 1]; } else lo = pts[6];
    }
    return 0.5 * (lo + hi);
  }

  /** max|S^1| - min|S^1| (the velocity-jump estimate of the lab; D independent). */
  async velocityJump() {
    const g = this.g;
    const enc = this.device.createCommandEncoder({ label: 'velocity jump' });
    g.red.run(enc, this.P.stVel, [g.terms[0]], this.size, 0, [OP.MIN, OP.MAX, OP.SUM, OP.SUM], { n: this.n }, 'velocity range');
    const r = await g.red.read(1, enc);
    return r[1] - r[0];
  }

  destroy() {
    for (const b of this.tensors.values()) b.destroy();
    this.tensors.clear();
    for (const b of Object.values(this.cbuf)) b.destroy();
    this.cbuf = {}; this.built = null;
  }
}
