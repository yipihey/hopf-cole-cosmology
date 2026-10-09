// state.js - the complete control state of the lab and its URL-hash encoding.
//
// The state object S uses readable names; the hash uses the short keys of
// SCHEMA. Only values that differ from the defaults are written, so a bare
// URL is the default experiment. Unknown / invalid hash entries are ignored.

export const GRID_2D = [128, 256, 512];
export const GRID_3D = [32, 48, 64, 96, 128];
export const METHODS = [0, 1, 2, 4];
export const OMEGA_M = [1, 0.3, 0.25, 0.4];     // 1 = Einstein-de Sitter (exact D^n growth)

export const SLOT_KINDS = [
  'sheetgpu', 'sheetcpu', 'sheetp1gpu', 'sheetp1cpu', 'sheetx', 'sheetxp1', 'nufft', 'cic', 'hc', 'hcdual', 'lin', 'phi', 'lnpsi', 'invj', 'lptsrc', 'lptcurl', 'fabs', 'fphase', 'psihat', 'speed',
];

export function defaultN(mode) { return mode === 3 ? 64 : (ENV.lite ? 128 : 256); }
/** Runtime environment, set once by the app after the WebGPU probe. */
export const ENV = { gpuCompute: false, lite: false };
/** Live update is on by default in 2D, and in 3D when the GPU compute path is active (it recomputes in ~0.1-0.3 s). */
export function defaultLive(mode, gc = true) { return mode !== 3 || (ENV.gpuCompute && gc); }

export function makeDefaults(mode = 2) {
  return {
    mode, n: defaultN(mode),
    // initial conditions
    ic: 'g',                // g = Gaussian random field, w = two plane waves, p = single peak
    shape: 'pl',            // pl = power law, bbks, eh
    pn: -1, ns: 1, gm: 30, seed: 1, R: 0.02, sg: 1,
    sp: 0,                  // sigma0 for the deterministic presets (0 = keep amplitudes)
    k1: [2, 0, 0], k2: [0, 3, 0], a1: 1, a2: 1, f1: 0, f2: 0,
    pa: 2, pw: 0.08,
    // dynamics
    D: 0.3, order: 2, hs: 'zel', nu: 1e-4, me: 1, mx: 30, live: defaultLive(mode, !ENV.lite),
    gc: !ENV.lite,          // 3D: WebGPU compute path (when available); off by default in lite mode
    perf: 'auto',           // performance preset: auto (decided after the GPU probe) | lite | full
    om: 1,                  // flat LCDM matter density (1 = EdS)
    // layout
    vis: 'fs',              // visible sections: f = fields, s = spectra, l = Legendre lab, p = PDFs, k = kernels, e = explain
    // 2D field panels: [kind, sub, cmap, log]  ('' = default)
    slots: [[ENV.lite ? 'sheetcpu' : 'sheetgpu', ENV.lite ? '' : 'both', '', ''], ['hc', '', '', ''], ['fabs', 'sheet', '', ''], ['fphase', 'sheet', '', '']],
    same: true, rmin: 0.1, rmax: 30,
    // 3D views
    v1: 'cic', vm: 'emission', vo: 10, s1: 'hc', sa: 2, si: 0.5, fo: 'cic',
    c1: 'inferno', c2: 'magma', c3: 'viridis', c4: 'twilight', l1: true, l2: true,
    // spectra: dm = modes per |k| bin of the direct (deposit-free) sheet spectra, rf = refinement of the NUFFT density (Fourier-refined map)
    dm: 128, rf: 2,
    // deconvolve the top-hat cell window of the cell-averaged estimators (exact sheet deposits, Hopf–Cole dual sheet) in P(k)
    dw: true,
    ser: mode === 3 ? ['lin', 'cic', 'sheetp1', 'hc', 'hcdual', 'spt'] : ['lin', 'sheet', 'sheetp1', 'hc', 'hcdual', 'spt'], km: 30,
    // PDFs: top-hat diameter in grid cells
    pd: 10,
    // Kernels lab: variable of the sweep plot (D | nu)
    kx: 'D',
    // Legendre lab: which sheet is the reference of the difference maps (plain | p1 = vertex-interpolated, mass-conserving | x, xp1 = exact clipped P0 / P1 | nu = NUFFT density of the refined map)
    sref: 'plain',
  };
}

const bool = (v) => (v ? '1' : '0');
const num = (v) => String(+Number(v).toPrecision(6));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// [stateKey, hashKey, type, extra]
const SCHEMA = [
  ['mode', 'm', 'enum', [2, 3]],
  ['n', 'n', 'int', [16, 512]],
  ['ic', 'ic', 'str', ['g', 'w', 'p']],
  ['shape', 'sh', 'str', ['pl', 'bbks', 'eh']],
  ['pn', 'pn', 'num', [-3, 2]],
  ['ns', 'ns', 'num', [-1, 2]],
  ['gm', 'gm', 'num', [0.5, 1000]],
  ['seed', 'sd', 'int', [0, 4294967295]],
  ['R', 'R', 'num', [0, 0.5]],
  ['sg', 'sg', 'num', [0, 10]],
  ['sp', 'sp', 'num', [0, 10]],
  ['k1', 'k1', 'ivec'],
  ['k2', 'k2', 'ivec'],
  ['a1', 'a1', 'num', [0, 10]],
  ['a2', 'a2', 'num', [0, 10]],
  ['f1', 'f1', 'num', [-720, 720]],
  ['f2', 'f2', 'num', [-720, 720]],
  ['pa', 'pa', 'num', [-20, 20]],
  ['pw', 'pw', 'num', [0.005, 0.5]],
  ['D', 'D', 'num', [0, 5]],
  ['order', 'o', 'int', [1, 4]],
  ['hs', 'hs', 'str', ['zel', 'lpt', 'lptT']],
  ['nu', 'nu', 'num', [1e-9, 1]],
  ['me', 'me', 'enum', METHODS],
  ['mx', 'mx', 'num', [1, 200]],
  ['live', 'lv', 'bool'],
  ['gc', 'gc', 'bool'],
  ['perf', 'perf', 'str', ['auto', 'lite', 'full']],
  ['om', 'om', 'omega'],
  ['vis', 'v', 'str'],
  ['slots', 'f', 'slots'],
  ['same', 'sr', 'bool'],
  ['rmin', 'r0', 'num', [1e-3, 1]],
  ['rmax', 'r1', 'num', [1, 1e4]],
  ['v1', 'v1', 'str', ['cic', 'sheet', 'sheetp1', 'sheetx', 'sheetxp1', 'nufft', 'hc', 'hcdual', 'lin']],
  ['vm', 'vm', 'str', ['mip', 'emission']],
  ['vo', 'vo', 'num', [0.1, 1000]],
  ['s1', 's1', 'str', ['cic', 'sheet', 'sheetp1', 'sheetx', 'sheetxp1', 'nufft', 'hc', 'hcdual', 'lin']],
  ['sa', 'sa', 'int', [0, 2]],
  ['si', 'si', 'num', [0, 1]],
  ['fo', 'fo', 'str', ['cic', 'sheet', 'sheetp1', 'sheetx', 'sheetxp1', 'nufft', 'hc', 'hcdual', 'lin']],
  ['c1', 'c1', 'str'], ['c2', 'c2', 'str'], ['c3', 'c3', 'str'], ['c4', 'c4', 'str'],
  ['l1', 'l1', 'bool'], ['l2', 'l2', 'bool'],
  ['ser', 'se', 'list'],
  ['km', 'km', 'num', [1, 5000]],
  ['dm', 'dm', 'int', [32, 512]],
  ['rf', 'rf', 'enum', [1, 2, 4]],
  ['dw', 'dw', 'bool'],
  ['pd', 'pd', 'int', [2, 64]],
  ['kx', 'kx', 'str', ['D', 'nu']],
  ['sref', 'lr', 'str', ['plain', 'p1', 'x', 'xp1', 'nu']],
];

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Hash string (without '#') for state S. */
export function encodeHash(S) {
  const D = makeDefaults(S.mode);
  const p = [];
  for (const [key, hk, type] of SCHEMA) {
    if (key === 'mode' && S.mode === 3) { p.push('m=3'); continue; }      // 2D is the default mode
    if (same(S[key], D[key])) continue;
    const v = S[key];
    let s;
    switch (type) {
      case 'bool': s = bool(v); break;
      case 'omega': s = num(v); break;
      case 'num': s = num(v); break;
      case 'ivec': s = v.join('_'); break;
      case 'slots': s = v.map((q) => q.join('.').replace(/\.+$/, '')).join('_'); break;
      case 'list': s = v.join('_') || '-'; break;
      default: s = String(v);
    }
    p.push(`${hk}=${encodeURIComponent(s)}`);
  }
  return p.join('&');
}

/** Parse a hash string (with or without leading '#') into a full state object. */
export function decodeHash(hash) {
  const h = hash.replace(/^#/, '');
  const q = new URLSearchParams(h);
  let mode = Number(q.get('m'));
  if (mode !== 3) mode = 2;
  const S = makeDefaults(mode);
  for (const [key, hk, type, extra] of SCHEMA) {
    if (!q.has(hk)) continue;
    const raw = q.get(hk);
    try {
      switch (type) {
        case 'enum': { const v = Number(raw); if (extra.includes(v)) S[key] = v; break; }
        case 'int': { const v = Math.round(Number(raw)); if (Number.isFinite(v)) S[key] = clamp(v, extra[0], extra[1]); break; }
        case 'num': { const v = Number(raw); if (Number.isFinite(v)) S[key] = clamp(v, extra[0], extra[1]); break; }
        case 'bool': S[key] = raw === '1' || raw === 'true'; break;
        case 'omega': { const v = Number(raw); const m = OMEGA_M.find((o) => Math.abs(o - v) < 1e-9); if (m !== undefined) S[key] = m; break; }
        case 'str': if (!extra || extra.includes(raw)) S[key] = raw; break;
        case 'ivec': {
          const v = raw.split('_').map((x) => Math.round(Number(x)));
          if (v.length >= 2 && v.every(Number.isFinite)) S[key] = [v[0], v[1], v[2] || 0];
          break;
        }
        case 'slots': {
          const parts = raw.split('_');
          parts.slice(0, 4).forEach((s, i) => {
            const f = s.split('.');
            if (SLOT_KINDS.includes(f[0])) S.slots[i] = [f[0], f[1] || '', f[2] || '', f[3] || ''];
          });
          break;
        }
        case 'list': S[key] = raw === '-' ? [] : raw.split('_').filter(Boolean); break;
        default: break;
      }
    } catch (e) { /* ignore malformed entries */ }
  }
  // consistency
  const grids = S.mode === 3 ? GRID_3D : GRID_2D;
  if (!grids.includes(S.n)) S.n = defaultN(S.mode);
  if (S.mode === 3 && S.me > 1) S.me = 1;
  if (!q.has('lv')) S.live = defaultLive(S.mode, S.gc);
  return S;
}
