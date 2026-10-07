// reduce.js - two-level GPU reductions (sum / min / max of up to four values per cell).
//
// Stage 1: a kernel (written by the user of this module, see RED_PRELUDE) evaluates a vec4<f32> per cell and
//          reduces it over its workgroup (256 threads) into partial[workgroup].
// Stage 2: red_final (one workgroup) reduces the partials into results[slot] (vec4<f32>).
// Each of the four channels has its own operation (OP.SUM / MIN / MAX), given in the uniform.
// The results of many reductions can be recorded into one command encoder and read back with ONE mapAsync.
//
// Stage-1 kernel contract (bindings): 0 = uniform RU, 1 = `part` (storage, read_write, provided by the prelude),
// 2.. = inputs.  A kernel looks like
//
//   @compute @workgroup_size(256)
//   fn my_kernel(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
//                @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
//     let i = linear_id(gid, nwg, 256u);
//     var v = ident(ru.ops);
//     if (i < ru.count) { v = ...; }
//     red_store(v, lid, wid, nwg);
//   }

import { ParamRing, makePipeline, makeBindGroup, dispatch1d, compile, readRegions, U_ENTRY, RO_ENTRY, RW_ENTRY, WGSL_LINEAR, STORAGE_RW } from './util.js';

export const OP = { SUM: 0, MIN: 1, MAX: 2 };

/** WGSL prelude shared by every stage-1 kernel module (declares ru at binding 0 and part at binding 1). */
export const RED_PRELUDE = /* wgsl */`
${WGSL_LINEAR}
struct RU { count: u32, n: u32, order: u32, x: u32, ops: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> ru: RU;
@group(0) @binding(1) var<storage, read_write> part: array<vec4<f32>>;
var<workgroup> shr: array<vec4<f32>, 256>;
fn ident(ops: vec4<u32>) -> vec4<f32> {
  var r = vec4<f32>(0.0);
  for (var c = 0; c < 4; c = c + 1) {
    if (ops[c] == 1u) { r[c] = 3.0e38; } else if (ops[c] == 2u) { r[c] = -3.0e38; }
  }
  return r;
}
fn combine(a: vec4<f32>, b: vec4<f32>, ops: vec4<u32>) -> vec4<f32> {
  var r = a;
  for (var c = 0; c < 4; c = c + 1) {
    if (ops[c] == 0u) { r[c] = a[c] + b[c]; } else if (ops[c] == 1u) { r[c] = min(a[c], b[c]); } else { r[c] = max(a[c], b[c]); }
  }
  return r;
}
fn red_store(v: vec4<f32>, lid: u32, wid: vec3<u32>, nwg: vec3<u32>) {
  shr[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { shr[lid] = combine(shr[lid], shr[lid + s], ru.ops); }
    workgroupBarrier();
  }
  if (lid == 0u) { part[wid.y * nwg.x + wid.x] = shr[0]; }
}
`;

const FINAL_WGSL = /* wgsl */`
struct FU { count: u32, slot: u32, p0: u32, p1: u32, ops: vec4<u32> };
@group(0) @binding(0) var<uniform> fu: FU;
@group(0) @binding(1) var<storage, read> pin: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> res: array<vec4<f32>>;
var<workgroup> shf: array<vec4<f32>, 256>;
fn ident(ops: vec4<u32>) -> vec4<f32> {
  var r = vec4<f32>(0.0);
  for (var c = 0; c < 4; c = c + 1) {
    if (ops[c] == 1u) { r[c] = 3.0e38; } else if (ops[c] == 2u) { r[c] = -3.0e38; }
  }
  return r;
}
fn combine(a: vec4<f32>, b: vec4<f32>, ops: vec4<u32>) -> vec4<f32> {
  var r = a;
  for (var c = 0; c < 4; c = c + 1) {
    if (ops[c] == 0u) { r[c] = a[c] + b[c]; } else if (ops[c] == 1u) { r[c] = min(a[c], b[c]); } else { r[c] = max(a[c], b[c]); }
  }
  return r;
}
@compute @workgroup_size(256)
fn red_final(@builtin(local_invocation_index) lid: u32) {
  var v = ident(fu.ops);
  var i = lid;
  loop {
    if (i >= fu.count) { break; }
    v = combine(v, pin[i], fu.ops);
    i = i + 256u;
  }
  shf[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { shf[lid] = combine(shf[lid], shf[lid + s], fu.ops); }
    workgroupBarrier();
  }
  if (lid == 0u) { res[fu.slot] = shf[0]; }
}
`;

const finalCache = new WeakMap();
function getFinal(device) {
  let p = finalCache.get(device);
  if (!p) {
    p = (async () => {
      const m = await compile(device, FINAL_WGSL, 'reduce final');
      return makePipeline(device, m, 'red_final', [U_ENTRY(0), RO_ENTRY(1), RW_ENTRY(2)], 'red_final');
    })();
    finalCache.set(device, p);
  }
  return p;
}

/** Layout entries of a stage-1 kernel with `nin` read-only inputs (bindings 2 ..). */
export function stage1Layout(nin) {
  const e = [U_ENTRY(0), RW_ENTRY(1)];
  for (let i = 0; i < nin; i++) e.push(RO_ENTRY(2 + i));
  return e;
}

export class Reducer {
  /** maxCount: the largest cell count ever reduced (sets the partials buffer). */
  constructor(device, maxCount, slots = 64) {
    this.device = device; this.ring = ParamRing.get(device);
    this.nPart = Math.ceil(maxCount / 256);
    this.slots = slots;
    this.part = device.createBuffer({ label: 'reduce partials', size: Math.max(16, this.nPart * 16), usage: STORAGE_RW });
    this.res = device.createBuffer({ label: 'reduce results', size: slots * 16, usage: STORAGE_RW });
    this.bgFinal = null;
  }
  async init() { this.F = await getFinal(this.device); this.bgFinal = makeBindGroup(this.device, this.F.bgl, [this.ring.resource(32), this.part, this.res], 'red final bg'); return this; }

  /**
   * Record: stage-1 kernel `pipe` (from makePipeline(..., stage1Layout(nin))) on `inputs` over `count` cells, then the final
   * reduction into results[slot].  ops: [op0..op3]; params: {n, order, x, f:[4]} written to the uniform.
   */
  run(enc, pipe, inputs, count, slot, ops, params = {}, label = 'reduce') {
    const { n = 0, order = 0, x = 0, f = [0, 0, 0, 0] } = params;
    const words = [['u', count], ['u', n], ['u', order], ['u', x], ['u', ops[0]], ['u', ops[1]], ['u', ops[2]], ['u', ops[3]], f[0], f[1], f[2], f[3]];
    const off = this.ring.write(words);
    const bg = makeBindGroup(this.device, pipe.bgl, [this.ring.resource(48), this.part, ...inputs], label);
    let pass = enc.beginComputePass({ label });
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bg, [off]);
    dispatch1d(pass, count, 256);
    pass.end();
    const nwg = Math.max(1, Math.ceil(count / 256));
    const off2 = this.ring.write([['u', nwg], ['u', slot], 0, 0, ['u', ops[0]], ['u', ops[1]], ['u', ops[2]], ['u', ops[3]]]);
    pass = enc.beginComputePass({ label: label + ' final' });
    pass.setPipeline(this.F.pipeline);
    pass.setBindGroup(0, this.bgFinal, [off2]);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  /** Read results[0 .. nslots-1] (4 floats each); appends the copy to `enc` (finished and submitted here). */
  async read(nslots, enc = null) {
    const [ab] = await readRegions(this.device, [{ buffer: this.res, bytes: nslots * 16 }], enc);
    return new Float32Array(ab);
  }
  destroy() { this.part.destroy(); this.res.destroy(); }
}
