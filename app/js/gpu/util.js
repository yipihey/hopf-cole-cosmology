// util.js - small WebGPU helpers shared by the compute modules (no dependencies except viz/gpu.js).
//
//  * compile():      shader module with error reporting
//  * ParamRing:      per-device rotating uniform buffer; every dispatch gets its own 256-byte slot
//                    (dynamic offsets), so many dispatches can be recorded into ONE command encoder
//  * readBuffer():   copy a storage buffer to a Float32Array / Uint32Array (pooled staging buffers)
//  * dispatch1d():   split a 1-D thread count into a (x, y) workgroup grid (limit 65535 per dimension)

import { compileModule } from '../viz/gpu.js';

export const GPU_USAGE = {
  STORAGE: 0x80 | 0x4 | 0x8,          // STORAGE | COPY_SRC | COPY_DST
};
// Literal WebGPU constants: the GPUBufferUsage global does not exist in browsers
// without WebGPU (e.g. Firefox on macOS/Linux), and referencing it at module load
// would break the whole module graph.  STORAGE = 0x80, COPY_SRC = 0x4, COPY_DST = 0x8.
export const STORAGE_RW = 0x80 | 0x4 | 0x8;

export async function compile(device, code, label) { return compileModule(device, code, label); }

/** WGSL helper: linear thread id from a 2-D dispatch of 1-D workgroups of size WG. */
export const WGSL_LINEAR = /* wgsl */`
fn linear_id(gid: vec3<u32>, nwg: vec3<u32>, wg: u32) -> u32 { return gid.y * (nwg.x * wg) + gid.x; }
`;

/** Dispatch ceil(count / wg) workgroups, folded into 2-D when above 65535. */
export function dispatch1d(pass, count, wg) {
  const groups = Math.max(1, Math.ceil(count / wg));
  if (groups <= 65535) pass.dispatchWorkgroups(groups, 1, 1);
  else {
    const x = 32768, y = Math.ceil(groups / x);
    pass.dispatchWorkgroups(x, y, 1);
  }
}

const RING_SLOTS = 8192, SLOT = 256;
const rings = new WeakMap();

export class ParamRing {
  static get(device) {
    let r = rings.get(device);
    if (!r) { r = new ParamRing(device); rings.set(device, r); }
    return r;
  }
  constructor(device) {
    this.device = device;
    this.buffer = device.createBuffer({ label: 'param ring', size: RING_SLOTS * SLOT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.next = 0;
    this.scratch = new ArrayBuffer(SLOT);
    this.u32 = new Uint32Array(this.scratch);
    this.f32 = new Float32Array(this.scratch);
  }
  /**
   * Write up to 64 32-bit words. Each entry is ['u', v] or ['f', v] or a bare number (f32).
   * Returns the dynamic offset of the slot.
   */
  write(words) {
    const off = (this.next++ % RING_SLOTS) * SLOT;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (Array.isArray(w)) { if (w[0] === 'u') this.u32[i] = w[1] >>> 0; else this.f32[i] = w[1]; }
      else this.f32[i] = w;
    }
    this.device.queue.writeBuffer(this.buffer, off, this.scratch, 0, 4 * Math.max(4, words.length));
    return off;
  }
  /** Binding resource for a uniform of `size` bytes with a dynamic offset. */
  resource(size = 64) { return { buffer: this.buffer, size }; }
}

/** Bind group layout entry helpers. */
export const U_ENTRY = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } });
export const RO_ENTRY = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
export const RW_ENTRY = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });

/** Compute pipeline with an explicit layout derived from (uniform?, #read-only, #read-write) in binding order. */
export function makePipeline(device, module, entryPoint, layoutEntries, label) {
  const bgl = device.createBindGroupLayout({ label: label + ' bgl', entries: layoutEntries });
  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint },
  });
  return { pipeline, bgl };
}

/** Bind group from a layout and an array of buffers (binding i = entries[i]); the uniform entry takes ring.resource(). */
export function makeBindGroup(device, bgl, resources, label) {
  return device.createBindGroup({
    label, layout: bgl,
    entries: resources.map((r, i) => ({ binding: i, resource: (r && r.buffer) ? r : { buffer: r } })),
  });
}

const stagingPool = new WeakMap();

/**
 * Read several buffer regions back to the CPU with ONE staging buffer, one submit and one mapAsync.
 * regions: [{buffer, bytes, offset?}] -> Promise<ArrayBuffer[]>. Pass an encoder to append the copies to
 * work already recorded in it (it is finished and submitted here).
 */
export async function readRegions(device, regions, encoder = null) {
  let total = 0;
  const offs = regions.map((r) => { const o = total; total += (r.bytes + 255) & ~255; return o; });
  let pool = stagingPool.get(device);
  if (!pool) { pool = []; stagingPool.set(device, pool); }
  const sizeClass = Math.max(256, 1 << Math.ceil(Math.log2(Math.max(4, total))));
  let st = null;
  const i = pool.findIndex((b) => b.size === sizeClass);
  if (i >= 0) st = pool.splice(i, 1)[0].buf;
  if (!st) st = device.createBuffer({ label: 'staging', size: sizeClass, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = encoder || device.createCommandEncoder();
  regions.forEach((r, j) => enc.copyBufferToBuffer(r.buffer, r.offset || 0, st, offs[j], (r.bytes + 3) & ~3));
  device.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ, 0, total);
  const mapped = st.getMappedRange(0, total);
  const out = regions.map((r, j) => mapped.slice(offs[j], offs[j] + r.bytes));
  st.unmap();
  pool.push({ size: sizeClass, buf: st });
  if (pool.length > 6) pool.shift().buf.destroy();
  return out;
}

export async function readBytes(device, buffer, byteLength, encoder = null) {
  return (await readRegions(device, [{ buffer, bytes: byteLength }], encoder))[0];
}
export async function readF32(device, buffer, count) { return new Float32Array(await readBytes(device, buffer, count * 4)); }
export async function readU32(device, buffer, count) { return new Uint32Array(await readBytes(device, buffer, count * 4)); }
