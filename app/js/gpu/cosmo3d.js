// cosmo3d.js - WebGPU compute path for the 3-D lab: LPT positions, CIC density, Hopf-Cole (log-domain)
// density, power spectra and Fourier maps, all on one shared GPUDevice.
//
// Conventions (identical to the WASM core): L = 1, scalar fields are f32 arrays of n^3 values with
// index (ix*n + iy)*n + iz, vector fields interleave [idx*3 + a], k_a = 2 pi m_a with
// m = i for i < n/2 and i - n otherwise.
//
// Public API (all buffers are GPUBuffers owned by the instance; the ones returned by positions(),
// cicDensity(), hopfCole(), uploadField() are PERSISTENT scratch buffers that are overwritten by the
// next call of the same method - read them back (readField) or consume them before then):
//
//   const g = new GpuCosmo3D(device, n);   await g.init();
//   await g.uploadTerms(terms /* Float32Array(3 n^3)[] */, orders /* number[] */);   (WASM-built terms) or
//   GpuLpt3D (lpt3d.js) builds the terms on the GPU and installs them with g.installTerms(bufs, orders)
//   await g.setPhi0(phi0 /* Float32Array(n^3), laplacian(phi) = delta0 */);   (GpuLpt3D.build writes it itself)
//   pos  = g.positions(gvals /* g_tau(D) per term */, order)  -> GPUBuffer 3 n^3 f32, unwrapped
//   disp = g.displacement(gvals, order [, enc, name])         -> GPUBuffer 3 n^3 f32, Psi = x - q (box units)
//   rho  = g.cicDensity(pos)                                  -> GPUBuffer n^3 f32, rho / rhobar
//   {delta, phi} = g.hopfCole(nu, D [, enc, src])             -> GPUBuffers n^3 f32 (delta, Phi_v = -2 nu ln psi); src = 'phi0' (Zel'dovich)
//                                                                or 'phi-eff' (nLPT effective potential, GpuLpt3D.legendre)
//   st = await g.potentialStats(name)                         -> {gmax, range} of a potential buffer, measured on the GPU
//   {k, p, n} = await g.powerSpectrum(field, nbins, deconvolveCic [, offset])      Float64Array each
//   {amp, phase} = await g.fourierMaps(field [, offset])      Float32Array n^2, fft-shifted [ikx*n + iky]
//   delta_s = await g.smoothTophat(field, radiusCells [, offset, dim])   Float32Array n^3: top-hat (sphere / disc) smoothed (field - offset)
//   an = await g.analyze(field, {nbins, offset, maps, cross}) -> one FFT, everything at once (see below)
//   ref = await g.setReference(delta0)                        FFT of a reference field (cross spectra)
//   arr = await g.readField(buf [, count])                    Float32Array
//   buf = g.uploadField(Float32Array)                         scratch field buffer
//   g.destroy()
//
// `offset` is subtracted from every cell before the FFT (use 1 for a rho/rhobar field).
//
// Numerical notes / limits
//  * CIC uses atomicAdd on u32 fixed-point weights, CIC_SCALE = 2^18 per unit mass.  A cell overflows
//    (wraps around) when it receives more than 2^32 / 2^18 = 16384 particle masses, i.e. rho/rhobar >= 16384;
//    quantisation is 3.8e-6 per weight and mass is conserved exactly (the 8th corner takes the remainder).
//  * Hopf-Cole and FFTs are single precision: the log-domain values carry |phi|/(2 nu) * 6e-8 absolute error,
//    which becomes ~ |phi_max| * 2e-3 in the 4th-order Hessian at 128^3; deltas agree with the f64 WASM result
//    to ~1e-4 of the field rms for the default parameters.
//  * Memory: 3 n^3 f32 per LPT term (13 raw terms for 4LPT: 312 MiB at 128^3), about 20 n^3 f32 of working buffers plus
//    the nLPT build/Legendre buffers of lpt3d.js (see its header; 688 MiB steady state at 128^3 4LPT).  The adapter's
//    maxStorageBufferBindingSize (128 MiB by default) limits n to 128 (positions: 12 n^3 bytes = 25 MB at 128^3,
//    201 MB at 256^3 > limit).  The constructor throws when a buffer would exceed the limits.

import { GpuFFT3D } from './fft3d.js';
import {
  ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, readRegions,
  U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW,
} from './util.js';
import { Reducer, OP, RED_PRELUDE, stage1Layout } from './reduce.js';

export const CIC_SCALE = 1 << 18;
const WG = 256;

const SHADERS = /* wgsl */`
${WGSL_LINEAR}

// ---------------------------------------------------------------- LPT positions
struct PosU { count: u32, n: u32, first: u32, qbase: u32, g: f32, dx: f32 };
@group(0) @binding(0) var<uniform> pu: PosU;
@group(0) @binding(1) var<storage, read> term: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos: array<f32>;
@compute @workgroup_size(256)
fn positions(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= pu.count) { return; }
  let n = pu.n;
  let iz = i % n; let iy = (i / n) % n; let ix = i / (n * n);
  var b = vec3<f32>(f32(ix), f32(iy), f32(iz)) * (pu.dx * f32(pu.qbase));
  if (pu.first == 0u) { b = vec3<f32>(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]); }
  let s = vec3<f32>(term[3u * i], term[3u * i + 1u], term[3u * i + 2u]);
  let r = b + pu.g * s;
  pos[3u * i] = r.x; pos[3u * i + 1u] = r.y; pos[3u * i + 2u] = r.z;
}

// ---------------------------------------------------------------- CIC
struct CicU { count: u32, n: u32, scale: u32, p0: u32, invdx: f32 };
@group(0) @binding(0) var<uniform> cu: CicU;
@group(0) @binding(1) var<storage, read> cpos: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@compute @workgroup_size(256)
fn cic_deposit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= cu.count) { return; }
  let mask = cu.n - 1u;
  let x = vec3<f32>(cpos[3u * i], cpos[3u * i + 1u], cpos[3u * i + 2u]) * cu.invdx - vec3<f32>(0.5);
  let fl = floor(x);
  let fr = x - fl;
  let i0 = vec3<u32>(bitcast<u32>(i32(fl.x)) & mask, bitcast<u32>(i32(fl.y)) & mask, bitcast<u32>(i32(fl.z)) & mask);
  var rem = cu.scale;
  let fs = f32(cu.scale);
  for (var c = 0u; c < 8u; c = c + 1u) {
    let bx = c & 1u; let by = (c >> 1u) & 1u; let bz = (c >> 2u) & 1u;
    let w = select(1.0 - fr.x, fr.x, bx == 1u) * select(1.0 - fr.y, fr.y, by == 1u) * select(1.0 - fr.z, fr.z, bz == 1u);
    var q = min(u32(w * fs + 0.5), rem);
    if (c == 7u) { q = rem; }
    rem = rem - q;
    if (q > 0u) {
      let idx = (((i0.x + bx) & mask) * cu.n + ((i0.y + by) & mask)) * cu.n + ((i0.z + bz) & mask);
      atomicAdd(&acc[idx], q);
    }
  }
}
@group(0) @binding(1) var<storage, read_write> acc2: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> rho: array<f32>;
@compute @workgroup_size(256)
fn cic_convert(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= cu.count) { return; }
  rho[i] = f32(atomicLoad(&acc2[i])) / f32(cu.scale);
}

// ---------------------------------------------------------------- Hopf-Cole (log domain)
struct HcU { count: u32, n: u32, stride: u32, w: u32, inv4nuD: f32, dx2: f32, s: f32, D: f32 };
@group(0) @binding(0) var<uniform> hu: HcU;
@group(0) @binding(1) var<storage, read> hin: array<f32>;
@group(0) @binding(2) var<storage, read_write> hout: array<f32>;

@compute @workgroup_size(256)
fn hc_scale_in(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= hu.count) { return; }
  hout[i] = hin[i] * hu.s;          // s = 1/(2 nu)  (a = phi/2nu)   or  -2 nu  (Phi_v = -2 nu ln psi)
}

@compute @workgroup_size(256)
fn hc_lse(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let e = linear_id(gid, nwg, 256u);
  if (e >= hu.count) { return; }
  let n = hu.n; let mask = n - 1u;
  let st = hu.stride;
  let x = (e / st) % n;
  let lb = e - x * st;
  let w = i32(hu.w);
  let xi = i32(x);
  let k = hu.inv4nuD * hu.dx2;
  var m = -3.0e38;
  for (var s = 0; s <= w; s = s + 1) {
    let c = -f32(s * s) * k;
    let yp = u32(xi + s) & mask;
    let ym = bitcast<u32>(xi - s) & mask;
    m = max(m, max(hin[lb + yp * st], hin[lb + ym * st]) + c);
  }
  var sum = 0.0;
  for (var s = 0; s <= w; s = s + 1) {
    let c = -f32(s * s) * k - m;
    let yp = u32(xi + s) & mask;
    sum = sum + exp(hin[lb + yp * st] + c);
    if (s > 0) {
      let ym = bitcast<u32>(xi - s) & mask;
      sum = sum + exp(hin[lb + ym * st] + c);
    }
  }
  hout[e] = m + log(sum);
}

// delta = det(I - D H) - 1, H the Hessian of phi (4th-order central differences, periodic)
fn fd_at(c: vec3<u32>, d: vec3<i32>) -> f32 {
  let n = hu.n; let m = i32(n - 1u);
  let a = u32((i32(c.x) + d.x) & m); let b = u32((i32(c.y) + d.y) & m); let cc = u32((i32(c.z) + d.z) & m);
  return hin[(a * n + b) * n + cc];
}
@compute @workgroup_size(256)
fn hc_delta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= hu.count) { return; }
  let n = hu.n;
  let c = vec3<u32>(i / (n * n), (i / n) % n, i % n);
  var wd = array<f32, 4>(1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0);
  var od = array<i32, 4>(-2, -1, 1, 2);
  let f0 = hin[i];
  var ex = array<vec3<i32>, 3>(vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1));
  let ih2 = 1.0 / hu.dx2;
  var h = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var a = 0; a < 3; a = a + 1) {
    // pure second derivative along a
    let fm2 = fd_at(c, ex[a] * -2); let fm1 = fd_at(c, -ex[a]);
    let fp1 = fd_at(c, ex[a]); let fp2 = fd_at(c, ex[a] * 2);
    h[a * 3 + a] = (-(fp2 + fm2) + 16.0 * (fp1 + fm1) - 30.0 * f0) * (ih2 / 12.0);
    for (var b = a + 1; b < 3; b = b + 1) {
      var s = 0.0;
      for (var p = 0; p < 4; p = p + 1) {
        for (var q = 0; q < 4; q = q + 1) {
          s = s + wd[p] * wd[q] * fd_at(c, ex[a] * od[p] + ex[b] * od[q]);
        }
      }
      h[a * 3 + b] = s * ih2; h[b * 3 + a] = s * ih2;
    }
  }
  let D = hu.D;
  let m00 = 1.0 - D * h[0]; let m01 = -D * h[1]; let m02 = -D * h[2];
  let m10 = -D * h[3]; let m11 = 1.0 - D * h[4]; let m12 = -D * h[5];
  let m20 = -D * h[6]; let m21 = -D * h[7]; let m22 = 1.0 - D * h[8];
  let det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
  hout[i] = det - 1.0;
}

// ---------------------------------------------------------------- top-hat smoothing (Fourier-space window, in place on the spectrum)
struct THU { count: u32, n: u32, dim: u32, p1: u32, r: f32 };
@group(0) @binding(0) var<uniform> thu: THU;
@group(0) @binding(1) var<storage, read_write> thf: array<vec2<f32>>;

// Bessel J1 (Numerical Recipes rational / asymptotic approximations; same as the CPU window in core/src/spectra.rs)
fn bessj1(x: f32) -> f32 {
  let ax = abs(x);
  if (ax < 8.0) {
    let y = x * x;
    let a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
    let a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return a1 / a2;
  }
  let z = 8.0 / ax;
  let y = z * z;
  let xx = ax - 2.356194491;
  let a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
  let a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  let ans = sqrt(0.636619772 / ax) * (cos(xx) * a1 - z * sin(xx) * a2);
  if (x < 0.0) { return -ans; }
  return ans;
}
// disc (dim 2): 2 J1(x)/x; sphere (dim 3): 3 (sin x - x cos x)/x^3 (series below x = 0.3: the closed form cancels in f32)
fn th_window(x: f32, dim: u32) -> f32 {
  if (x < 1e-6) { return 1.0; }
  if (dim == 2u) { return 2.0 * bessj1(x) / x; }
  if (x < 0.3) { let x2 = x * x; return 1.0 - x2 / 10.0 + x2 * x2 / 280.0; }
  return 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
}
@compute @workgroup_size(256)
fn tophat(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  if (i >= thu.count) { return; }
  let n = thu.n;
  let mx = f32(fftfreq(i / (n * n), n)); let my = f32(fftfreq((i / n) % n, n)); let mz = f32(fftfreq(i % n, n));
  let k = 6.283185307179586 * sqrt(mx * mx + my * my + mz * mz);
  thf[i] = thf[i] * th_window(k * thu.r, thu.dim);
}

// ---------------------------------------------------------------- spectra
struct BinU { n: u32, deconv: u32, p0: u32, p1: u32, norm: f32 };
@group(0) @binding(0) var<uniform> bu: BinU;
@group(0) @binding(1) var<storage, read> fh: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> rf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;
@group(0) @binding(4) var<storage, read> chunks: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> partial: array<vec4<f32>>;
var<workgroup> sh: array<vec3<f32>, 256>;

fn sinc2(m: i32, n: u32) -> f32 {
  let x = 3.14159265358979 * f32(m) / f32(n);
  if (abs(x) < 1e-6) { return 1.0; }
  let s = sin(x) / x;
  return s * s;
}
fn fftfreq(i: u32, n: u32) -> i32 { if (i < n / 2u) { return i32(i); } return i32(i) - i32(n); }

@compute @workgroup_size(256)
fn bin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let ch = chunks[wid.x];
  let n = bu.n;
  var a = vec3<f32>(0.0);
  var i = ch.x + lid;
  loop {
    if (i >= ch.y) { break; }
    let e = perm[i];
    let v = fh[e]; let g = rf[e];
    let plain = dot(v, v) * bu.norm;
    var dec = plain;
    if (bu.deconv != 0u) {
      let w = sinc2(fftfreq(e / (n * n), n), n) * sinc2(fftfreq((e / n) % n, n), n) * sinc2(fftfreq(e % n, n), n);
      dec = plain / (w * w);
    }
    a = a + vec3<f32>(dec, plain, (v.x * g.x + v.y * g.y) * bu.norm);
    i = i + 256u;
  }
  sh[lid] = a;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { sh[lid] = sh[lid] + sh[lid + s]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partial[wid.x] = vec4<f32>(sh[0], 0.0); }
}

// max |f^|^2 over all cells -> bits of the (non-negative) float in a u32 (atomicMax is monotone)
struct MaxU { count: u32 };
@group(0) @binding(0) var<uniform> mu: MaxU;
@group(0) @binding(1) var<storage, read> mf: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> mx: array<atomic<u32>>;
var<workgroup> shm: array<f32, 256>;
@compute @workgroup_size(256)
fn maxabs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = linear_id(gid, nwg, 256u);
  var v = 0.0;
  if (i < mu.count) { let c = mf[i]; v = dot(c, c); }
  shm[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { shm[lid] = max(shm[lid], shm[lid + s]); }
    workgroupBarrier();
  }
  if (lid == 0u) { atomicMax(&mx[0], bitcast<u32>(shm[0])); }
}

struct MapU { n: u32 };
@group(0) @binding(0) var<uniform> pu2: MapU;
@group(0) @binding(1) var<storage, read> gf: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> gmx: array<u32>;
@group(0) @binding(3) var<storage, read_write> gout: array<f32>;
@compute @workgroup_size(256)
fn maps(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = linear_id(gid, nwg, 256u);
  let n = pu2.n;
  if (t >= n * n) { return; }
  let i = t / n; let j = t % n;
  let fi = (i + n / 2u) % n; let fj = (j + n / 2u) % n;
  let v = gf[(fi * n + fj) * n];
  let m2 = max(bitcast<f32>(gmx[0]), 1e-30);
  let r = sqrt(dot(v, v) / m2);
  gout[t] = log(max(r, 1e-30)) * 0.4342944819032518;
  gout[n * n + t] = atan2(v.y, v.x);
}
`;

// min / max / max|grad|^2 of a potential (4th-order central differences, periodic): the Hopf-Cole window statistics
const STATS_SHADERS = /* wgsl */`
${RED_PRELUDE}
@group(0) @binding(2) var<storage, read> fld: array<f32>;
fn at3(x: i32, y: i32, z: i32, n: u32) -> f32 {
  let m = i32(n) - 1;
  return fld[(u32(x & m) * n + u32(y & m)) * n + u32(z & m)];
}
@compute @workgroup_size(256)
fn st_phi(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = linear_id(gid, nwg, 256u);
  var v = ident(ru.ops);
  if (i < ru.count) {
    let n = ru.n;
    let x = i32(i / (n * n)); let y = i32((i / n) % n); let z = i32(i % n);
    let s = f32(n) / 12.0;
    let gx = (8.0 * (at3(x + 1, y, z, n) - at3(x - 1, y, z, n)) - (at3(x + 2, y, z, n) - at3(x - 2, y, z, n))) * s;
    let gy = (8.0 * (at3(x, y + 1, z, n) - at3(x, y - 1, z, n)) - (at3(x, y + 2, z, n) - at3(x, y - 2, z, n))) * s;
    let gz = (8.0 * (at3(x, y, z + 1, n) - at3(x, y, z - 1, n)) - (at3(x, y, z + 2, n) - at3(x, y, z - 2, n))) * s;
    let f = fld[i];
    v = vec4<f32>(f, f, gx * gx + gy * gy + gz * gz, 0.0);
  }
  red_store(v, lid, wid, nwg);
}
`;

const pipeCache = new WeakMap();
function getPipelines(device) {
  let p = pipeCache.get(device);
  if (!p) {
    p = (async () => {
      const m = await compile(device, SHADERS, 'cosmo3d');
      const ms = await compile(device, STATS_SHADERS, 'cosmo3d stats');
      const U = U_ENTRY, R = RO_ENTRY, W = RW_ENTRY;
      const mk = (name, entries) => makePipeline(device, m, name, entries, name);
      return {
        positions: mk('positions', [U(0), R(1), W(2)]),
        cicDeposit: mk('cic_deposit', [U(0), R(1), W(2)]),
        cicConvert: mk('cic_convert', [U(0), W(1), W(2)]),
        hcScale: mk('hc_scale_in', [U(0), R(1), W(2)]),
        hcLse: mk('hc_lse', [U(0), R(1), W(2)]),
        hcDelta: mk('hc_delta', [U(0), R(1), W(2)]),
        bin: mk('bin', [U(0), R(1), R(2), R(3), R(4), W(5)]),
        maxabs: mk('maxabs', [U(0), R(1), W(2)]),
        maps: mk('maps', [U(0), R(1), R(2), W(3)]),
        tophat: mk('tophat', [U(0), W(1)]),
        stPhi: makePipeline(device, ms, 'st_phi', stage1Layout(1), 'st_phi'),
      };
    })();
    pipeCache.set(device, p);
  }
  return p;
}

const CHUNK = 2048;

export class GpuCosmo3D {
  constructor(device, n) {
    if (!Number.isInteger(Math.log2(n)) || n < 8) throw new Error('GpuCosmo3D: n must be a power of two >= 8');
    this.device = device; this.n = n; this.size = n * n * n;
    this.fbytes = 4 * this.size;
    const lim = device.limits;
    if (3 * this.fbytes > lim.maxStorageBufferBindingSize || 3 * this.fbytes > lim.maxBufferSize) {
      throw new Error(`GpuCosmo3D: the ${n}^3 position buffer (${(3 * this.fbytes / 1048576).toFixed(0)} MiB) exceeds the adapter limit maxStorageBufferBindingSize = ${(lim.maxStorageBufferBindingSize / 1048576).toFixed(0)} MiB`);
    }
    this.ring = ParamRing.get(device);
    this.fft = new GpuFFT3D(device, n);
    this.terms = []; this.orders = [];
    this.bufs = new Map();
    this.bgs = new Map();
    this.binTables = new Map();
    this.gmax = 0; this.phiRange = 0;
    this.pstat = {};                         // potential name -> {gmax, range}: window statistics of 'phi0', 'phi-eff'
    this.hasPhi0 = false; this.hasRef = false;
    this.red = new Reducer(device, n * n * n);
    this.destroyed = false;
    this.lastTimings = {};
  }

  async init() {
    this.P = await getPipelines(this.device);
    await this.fft.init();
    await this.red.init();
    return this;
  }

  // ------------------------------------------------------------------ buffer management
  buf(name, bytes) {
    let b = this.bufs.get(name);
    if (!b) {
      b = this.device.createBuffer({ label: name, size: Math.max(16, bytes), usage: STORAGE_RW });
      this.bufs.set(name, b);
    }
    return b;
  }
  bg(key, pipe, resources) {
    let g = this.bgs.get(key);
    if (!g) { g = makeBindGroup(this.device, pipe.bgl, resources, key); this.bgs.set(key, g); }
    return g;
  }
  /** Record one 1-D compute dispatch with a ring-allocated uniform. */
  _dispatch(enc, name, pipe, bindGroup, words, count, passLabel) {
    const slot = this.ring.write(words);
    const pass = enc.beginComputePass({ label: passLabel || name });
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bindGroup, [slot]);
    dispatch1d(pass, count, WG);
    pass.end();
  }
  _submit(enc) { this.device.queue.submit([enc.finish()]); }

  // ------------------------------------------------------------------ uploads
  /** Drop the bind groups that reference term buffers (they are rebuilt lazily). */
  _dropTermBindGroups() {
    for (const k of [...this.bgs.keys()]) if (/^(pos|disp|sc)/.test(k)) this.bgs.delete(k);
  }
  /** Upload the LPT term fields (interleaved xyz, 3 n^3 floats each) and their orders. Replaces previous terms. */
  async uploadTerms(terms, orders) {
    const d = this.device;
    for (const b of this.terms) b.destroy();
    this._dropTermBindGroups();
    this.terms = []; this.orders = orders.slice();
    d.pushErrorScope('out-of-memory');
    d.pushErrorScope('validation');
    for (let t = 0; t < terms.length; t++) {
      if (terms[t].length !== 3 * this.size) throw new Error(`uploadTerms: term ${t} has ${terms[t].length} values, expected ${3 * this.size}`);
      const b = d.createBuffer({ label: 'term' + t, size: this.fbytes * 3, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, terms[t]);
      this.terms.push(b);
    }
    const ve = await d.popErrorScope();
    const oe = await d.popErrorScope();
    if (ve || oe) throw new Error('GPU buffer allocation failed for the LPT terms: ' + (oe ? oe.message : ve.message));
  }
  /** Take ownership of term buffers built on the GPU (GpuLpt3D.build); the previous ones are destroyed. */
  installTerms(bufs, orders) {
    for (const b of this.terms) b.destroy();
    this._dropTermBindGroups();
    this.terms = bufs.slice(); this.orders = orders.slice();
  }

  /**
   * Upload the Lagrangian potential (laplacian phi = delta0) into the 'phi0' source and measure its window statistics on the
   * GPU (max |grad phi| with 4th-order differences, value range).  Async: the statistics need one small readback.
   */
  async setPhi0(phi) {
    if (phi.length !== this.size) throw new Error('setPhi0: wrong length');
    this.device.queue.writeBuffer(this.buf('phi0', this.fbytes), 0, phi);
    await this.potentialStats('phi0');
  }

  /** Measure {gmax, range} of the potential held in the named scratch buffer ('phi0' | 'phi-eff') and register it. */
  async potentialStats(name, encoder = null) {
    const enc = encoder || this.device.createCommandEncoder({ label: 'potential stats' });
    this.red.run(enc, this.P.stPhi, [this.buf(name, this.fbytes)], this.size, 0, [OP.MIN, OP.MAX, OP.MAX, OP.SUM], { n: this.n }, 'phi stats');
    const r = await this.red.read(1, enc);
    this.setPotentialStats(name, r);
    return this.pstat[name];
  }
  /** Register the statistics from a reduction result [lo, hi, max|grad|^2, _]. */
  setPotentialStats(name, r, off = 0) {
    const st = { gmax: Math.sqrt(Math.max(0, r[off + 2])) * 1.02, range: r[off + 1] - r[off] };    // small margin: the CPU reference uses spectral gradients
    this.pstat[name] = st;
    if (name === 'phi0') { this.gmax = st.gmax; this.phiRange = st.range; this.hasPhi0 = true; }
    return st;
  }

  /** Upload a CPU field into the shared scratch field buffer and return it. */
  uploadField(arr) {
    if (arr.length !== this.size) throw new Error('uploadField: wrong length');
    const b = this.buf('field-upload', this.fbytes);
    this.device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  async readField(buffer, count = this.size) {
    const [ab] = await readRegions(this.device, [{ buffer, bytes: count * 4 }]);
    return new Float32Array(ab);
  }
  /** Wait until all submitted GPU work has finished (for timing). */
  async sync() { await this.device.queue.onSubmittedWorkDone(); }

  // ------------------------------------------------------------------ positions, CIC
  /**
   * x = q + sum_t g_t S^t over the terms with order <= `order` (q = grid coordinates, unwrapped).
   * gvals[t] is g_t(D) for ALL uploaded terms (WASM: term_g(D); GPU-built terms: term_g_unmerged(D)).
   */
  positions(gvals, order, encoder = null) { return this._sumTerms('pos', gvals, order, 1, encoder); }
  /** Psi = sum_t g_t S^t (displacement, box units, interleaved xyz) into the persistent buffer 'disp'. */
  displacement(gvals, order, encoder = null, name = 'disp') { return this._sumTerms(name, gvals, order, 0, encoder); }
  _sumTerms(name, gvals, order, qbase, encoder) {
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: name });
    const out = this.buf(name, 3 * this.fbytes);
    let first = true;
    for (let t = 0; t < this.terms.length; t++) {
      if (this.orders[t] > order) continue;
      const g = this.bg(name + t, this.P.positions, [this.ring.resource(32), this.terms[t], out]);
      this._dispatch(enc, name, this.P.positions, g, [['u', this.size], ['u', this.n], ['u', first ? 1 : 0], ['u', qbase], gvals[t], 1 / this.n], this.size, name + ' t' + t);
      first = false;
    }
    if (first) throw new Error('positions: no LPT term with order <= ' + order);
    if (own) this._submit(enc);
    return out;
  }

  /** CIC deposit of n^3 unit-mass particles; returns rho/rhobar (mean exactly 1). Positions are wrapped here. */
  cicDensity(posBuffer, encoder = null) {
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'cic' });
    const acc = this.buf('cic-acc', this.fbytes), rho = this.buf('rho', this.fbytes);
    enc.clearBuffer(acc);
    const words = [['u', this.size], ['u', this.n], ['u', CIC_SCALE], 0, this.n];
    const g1 = makeBindGroup(this.device, this.P.cicDeposit.bgl, [this.ring.resource(32), posBuffer, acc], 'cic dep');
    this._dispatch(enc, 'cic', this.P.cicDeposit, g1, words, this.size, 'cic deposit');
    const g2 = this.bg('cicconv', this.P.cicConvert, [this.ring.resource(32), acc, rho]);
    this._dispatch(enc, 'cic', this.P.cicConvert, g2, words, this.size, 'cic convert');
    if (own) this._submit(enc);
    return rho;
  }

  // ------------------------------------------------------------------ Hopf-Cole
  /** Window half-width (cells) used for the given nu, D and potential source. */
  hcWindow(nu, D, src = 'phi0') {
    const reach = D * this.pstat[src].gmax + 6 * Math.sqrt(2 * nu * D);
    return Math.min(this.n / 2, Math.ceil(reach * this.n) + 2);
  }
  /** Exponent range (phi_max - phi_min)/(2 nu), the analogue of hc_exponent_range. */
  hcExponentRange(nu, src = 'phi0') { return this.pstat[src].range / (2 * nu); }

  /**
   * Real-space log-domain Hopf-Cole solve (refine = 1).  Returns {delta, phi, w} with delta = det(I - D H) - 1,
   * H = Hessian of phi = Phi_v = -2 nu ln psi (up to an additive constant).
   * `src` names the potential: 'phi0' (Zel'dovich, setPhi0) or 'phi-eff' (nLPT effective potential, GpuLpt3D.legendre);
   * its window statistics must be registered (setPhi0 / potentialStats).
   */
  hopfCole(nu, D, encoder = null, src = 'phi0') {
    if (!this.pstat[src]) throw new Error(`hopfCole: potential '${src}' has no statistics (call setPhi0 / potentialStats first)`);
    const own = !encoder;
    const enc = encoder || this.device.createCommandEncoder({ label: 'hopf-cole' });
    const n = this.n, N = this.size, dx = 1 / n;
    const A = this.buf('hcA', this.fbytes), B = this.buf('hcB', this.fbytes);
    const phi = this.buf('hc-phi', this.fbytes), delta = this.buf('hc-delta', this.fbytes);
    const phi0 = this.buf(src, this.fbytes);
    const w = this.hcWindow(nu, D, src);
    const u = (stride, s) => [['u', N], ['u', n], ['u', stride], ['u', w], 1 / (4 * nu * D), dx * dx, s, D];
    const R = this.ring.resource(32);
    // a = phi0 / (2 nu)
    this._dispatch(enc, 'hc', this.P.hcScale, this.bg('hc-in-' + src, this.P.hcScale, [R, phi0, A]), u(1, 1 / (2 * nu)), N, 'hc a');
    let cur = A, nxt = B;
    if (nu * D > 0) {
      for (const stride of [n * n, n, 1]) {
        const g = this.bg('hc-lse' + (cur === A ? 'AB' : 'BA'), this.P.hcLse, [R, cur, nxt]);
        this._dispatch(enc, 'hc', this.P.hcLse, g, u(stride, 0), N, 'hc lse');
        [cur, nxt] = [nxt, cur];
      }
    }
    // Phi_v = -2 nu ln psi
    this._dispatch(enc, 'hc', this.P.hcScale, this.bg('hc-out' + (cur === A ? 'A' : 'B'), this.P.hcScale, [R, cur, phi]), u(1, -2 * nu), N, 'hc phi');
    this._dispatch(enc, 'hc', this.P.hcDelta, this.bg('hc-fd', this.P.hcDelta, [R, phi, delta]), u(1, 0), N, 'hc delta');
    if (own) this._submit(enc);
    return { delta, phi, w };
  }

  // ------------------------------------------------------------------ spectra
  /** Mode -> bin tables (cached per nbins): sorted mode permutation, chunk list, k sums and counts. */
  _bins(nbins) {
    let T = this.binTables.get(nbins);
    if (T) return T;
    const n = this.n, N = this.size;
    const kf = 2 * Math.PI, knyq = Math.PI * n;
    const kmin = kf * 0.9, kmax = knyq * Math.sqrt(3) * 1.01;
    const edge0 = kmin, edgeN = kmin * Math.pow(kmax / kmin, 1);
    const lk0 = Math.log(edge0), dlk = Math.log(edgeN / edge0) / nbins;
    const maxm2 = 3 * (n / 2) * (n / 2);
    const binOfM2 = new Int16Array(maxm2 + 1).fill(-1);
    const kOfM2 = new Float64Array(maxm2 + 1);
    for (let m2 = 1; m2 <= maxm2; m2++) {
      const k = Math.sqrt(m2 * (kf * kf));
      kOfM2[m2] = k;
      if (k < edge0 || k >= edgeN) continue;
      binOfM2[m2] = Math.min(nbins - 1, Math.floor((Math.log(k) - lk0) / dlk));
    }
    const m2f = new Int32Array(n);
    for (let i = 0; i < n; i++) { const m = i < n / 2 ? i : i - n; m2f[i] = m * m; }
    const binIdx = new Int16Array(N);
    const cnt = new Float64Array(nbins), ksum = new Float64Array(nbins);
    let p = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const a = m2f[i] + m2f[j];
      for (let k = 0; k < n; k++, p++) {
        const m2 = a + m2f[k];
        const b = binOfM2[m2];
        binIdx[p] = b;
        if (b >= 0) { cnt[b]++; ksum[b] += kOfM2[m2]; }
      }
    }
    const offs = new Uint32Array(nbins + 1);
    for (let b = 0; b < nbins; b++) offs[b + 1] = offs[b] + cnt[b];
    const fill = offs.slice(0, nbins);
    const perm = new Uint32Array(offs[nbins]);
    for (let q = 0; q < N; q++) { const b = binIdx[q]; if (b >= 0) perm[fill[b]++] = q; }
    const chunks = [];
    const chunkBin = [];
    for (let b = 0; b < nbins; b++) {
      for (let s = offs[b]; s < offs[b + 1]; s += CHUNK) { chunks.push(s, Math.min(offs[b + 1], s + CHUNK), b, 0); chunkBin.push(b); }
    }
    const d = this.device;
    const permBuf = d.createBuffer({ label: 'bin perm', size: Math.max(16, perm.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(permBuf, 0, perm);
    const chunkArr = new Uint32Array(chunks);
    const chunkBuf = d.createBuffer({ label: 'bin chunks', size: Math.max(16, chunkArr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(chunkBuf, 0, chunkArr);
    const nchunk = chunkBin.length;
    const partial = d.createBuffer({ label: 'bin partial', size: Math.max(16, nchunk * 16), usage: STORAGE_RW });
    T = { permBuf, chunkBuf, partial, nchunk, chunkBin, cnt, ksum, nbins };
    this.binTables.set(nbins, T);
    return T;
  }

  /**
   * One forward FFT of (field - offset) and everything derived from it.
   * opts: {nbins = 40, offset = 0, maps = false, cross = false}.
   * Returns {plain, dec, cross, maps}:
   *   plain / dec: {k, p, n} (Float64Array) binned P(k) = L^3 |f^|^2 / N^2 (dec divides by the CIC window^2),
   *   cross: {k, p, n} of Re[f^ g^*] with the reference field of setReference() (or null),
   *   maps: {amp, phase} Float32Array n^2 (fft-shifted [ikx*n+iky], kz = 0 plane, amp = log10|f^|/max).
   */
  async analyze(field, opts = {}) {
    const { nbins = 40, offset = 0, maps = false, cross = false } = opts;
    if (cross && !this.hasRef) throw new Error('analyze: cross spectrum requested but no reference set');
    const n = this.n, N = this.size;
    const T = this._bins(nbins);
    const cA = this.buf('cplx-a', 2 * this.fbytes), fh = this.buf('fhat', 2 * this.fbytes);
    const refBuf = this.hasRef ? this.buf('fhat-ref', 2 * this.fbytes) : fh;
    const enc = this.device.createCommandEncoder({ label: 'analyze' });
    this.fft.packReal(field, cA, enc, offset);
    this.fft.forward(cA, fh, enc);
    // binned sums
    const slot = this.ring.write([['u', n], ['u', 1], 0, 0, 1 / (N * N)]);
    const slotDec = slot;
    const pass = enc.beginComputePass({ label: 'bin' });
    pass.setPipeline(this.P.bin.pipeline);
    pass.setBindGroup(0, makeBindGroup(this.device, this.P.bin.bgl, [this.ring.resource(32), fh, refBuf, T.permBuf, T.chunkBuf, T.partial], 'bin'), [slotDec]);
    pass.dispatchWorkgroups(T.nchunk);
    pass.end();
    const regions = [{ buffer: T.partial, bytes: T.nchunk * 16 }];
    if (maps) {
      const mx = this.buf('fmax', 16), out = this.buf('fmap-out', 2 * n * n * 4);
      enc.clearBuffer(mx);
      this._dispatch(enc, 'max', this.P.maxabs, makeBindGroup(this.device, this.P.maxabs.bgl, [this.ring.resource(16), fh, mx], 'max'), [['u', N]], N, 'max |f|');
      this._dispatch(enc, 'maps', this.P.maps, makeBindGroup(this.device, this.P.maps.bgl, [this.ring.resource(16), fh, mx, out], 'maps'), [['u', n]], n * n, 'fourier maps');
      regions.push({ buffer: out, bytes: 2 * n * n * 4 });
    }
    const res = await readRegions(this.device, regions, enc);
    const part = new Float32Array(res[0]);
    // The kernel always deconvolves when bu.deconv = 1, so the 'dec' channel is dec and 'plain' is plain.
    const sums = [new Float64Array(nbins), new Float64Array(nbins), new Float64Array(nbins)];
    for (let c = 0; c < T.nchunk; c++) {
      const b = T.chunkBin[c];
      sums[0][b] += part[4 * c]; sums[1][b] += part[4 * c + 1]; sums[2][b] += part[4 * c + 2];
    }
    const pack = (s) => {
      let m = 0;
      for (let b = 0; b < nbins; b++) if (T.cnt[b] > 0) m++;
      const k = new Float64Array(m), p = new Float64Array(m), nn = new Float64Array(m);
      let j = 0;
      for (let b = 0; b < nbins; b++) if (T.cnt[b] > 0) { k[j] = T.ksum[b] / T.cnt[b]; p[j] = s[b] / T.cnt[b]; nn[j] = T.cnt[b]; j++; }
      return { k, p, n: nn };
    };
    const out = { dec: pack(sums[0]), plain: pack(sums[1]), cross: cross ? pack(sums[2]) : null, maps: null };
    if (maps) {
      const mm = new Float32Array(res[1]);
      out.maps = { amp: mm.slice(0, n * n), phase: mm.slice(n * n) };
    }
    return out;
  }

  /**
   * Top-hat (sphere; dim = 2: disc) smoothing of `field - offset`: forward FFT, multiply the spectrum by the Fourier window
   * W(kR) (sphere 3 [sin x - x cos x]/x^3, disc 2 J1(x)/x, W(0) = 1, x = |k| R, k = 2 pi m with the fftfreq integers m, L = 1,
   * R = radiusCells / n), inverse FFT, read back.  Returns the smoothed (field - offset) as a Float32Array(n^3), e.g. the smoothed
   * overdensity of a density buffer with offset 1.  Uses the shared FFT scratch ('cplx-a', 'fhat'): call it from the serialised GPU queue.
   */
  async smoothTophat(field, radiusCells, offset = 0, dim = 3) {
    const n = this.n, N = this.size, d = this.device;
    const cA = this.buf('cplx-a', 2 * this.fbytes), fh = this.buf('fhat', 2 * this.fbytes), out = this.buf('th-out', this.fbytes);
    const enc = d.createCommandEncoder({ label: 'tophat smoothing' });
    this.fft.packReal(field, cA, enc, offset);
    this.fft.forward(cA, fh, enc);
    this._dispatch(enc, 'tophat', this.P.tophat, this.bg('tophat', this.P.tophat, [this.ring.resource(32), fh]),
      [['u', N], ['u', n], ['u', dim], ['u', 0], radiusCells / n], N, 'tophat window');
    this.fft.inverse(fh, cA, enc);
    this.fft.unpackReal(cA, out, enc);
    const [ab] = await readRegions(d, [{ buffer: out, bytes: this.fbytes }], enc);
    return new Float32Array(ab);
  }

  /** Binned P(k) of `field - offset`, same convention as CosmoSim.power_spectrum. */
  async powerSpectrum(field, nbins, deconvolveCic, offset = 0) {
    const a = await this.analyze(field, { nbins, offset });
    return deconvolveCic ? a.dec : a.plain;
  }
  /** log10|f^|/max and arg f^ in the kz = 0 plane, fft-shifted [ikx*n + iky]. */
  async fourierMaps(field, offset = 0) {
    const a = await this.analyze(field, { nbins: 8, offset, maps: true });
    return a.maps;
  }
  /** FFT a reference field once (e.g. delta0) so that analyze({cross: true}) can correlate with it. */
  async setReference(arr, nbins = 40) {
    const b = this.uploadField(arr);
    const fhRef = this.buf('fhat-ref', 2 * this.fbytes);
    const cA = this.buf('cplx-a', 2 * this.fbytes);
    const enc = this.device.createCommandEncoder({ label: 'reference' });
    this.fft.packReal(b, cA, enc, 0);
    this.fft.forward(cA, fhRef, enc);
    this._submit(enc);
    this.hasRef = true;
    return this.analyze(b, { nbins, maps: true });
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of this.terms) b.destroy();
    for (const b of this.bufs.values()) b.destroy();
    for (const T of this.binTables.values()) { T.permBuf.destroy(); T.chunkBuf.destroy(); T.partial.destroy(); }
    this.terms = []; this.bufs.clear(); this.bgs.clear(); this.binTables.clear();
    this.fft.destroy(); this.red.destroy();
  }
}
