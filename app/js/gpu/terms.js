// terms.js - extract the LPT term fields of a CosmoSim for GpuCosmo3D.uploadTerms.
//
// Uses the term_psi / term_orders / term_g accessors of the WASM core.  If the loaded package predates
// them, falls back to the per-order shape fields lpt_psi(order, comp), which are exact in Einstein-de Sitter
// (g_tau(D) = D^order * lpt coefficient) - the default cosmology of the lab.

/** @returns {{terms: Float32Array[], orders: number[], gvals: (D:number)=>number[], source: string}} */
export function getTerms(sim, n) {
  const N = n * n * n;
  const inter = (f) => {
    const o = new Float32Array(3 * N);
    for (let c = 0; c < 3; c++) { const a = f(c); for (let i = 0; i < N; i++) o[3 * i + c] = a[i]; }
    return o;
  };
  if (typeof sim.term_psi === 'function' && typeof sim.term_orders === 'function' && typeof sim.term_g === 'function') {
    const orders = Array.from(sim.term_orders());
    return { terms: orders.map((_, t) => inter((c) => sim.term_psi(t, c))), orders, gvals: (D) => Array.from(sim.term_g(D)), source: 'term_psi' };
  }
  const bo = sim.built_order();
  const orders = []; const terms = [];
  for (let o = 1; o <= bo; o++) { orders.push(o); terms.push(inter((c) => sim.lpt_psi(o, c))); }
  return { terms, orders, gvals: (D) => orders.map((o) => Math.pow(D, o)), source: 'lpt_psi (EdS fallback)' };
}

/** g_tau(D) for every term (same indexing as the uploaded terms). */
export function termGrowth(sim, orders, D) {
  if (typeof sim.term_g === 'function') return Array.from(sim.term_g(D));
  return orders.map((o) => Math.pow(D, o));
}

// ---------------------------------------------------------------------------------------------------------------
// Unmerged (raw recursion) term list, used by the GPU nLPT build (lpt3d.js).  The spatial fields S^tau of the raw list
// do not depend on the cosmology; only g_tau(D) does.  In EdS g_tau = c_tau D^order (merging terms of equal order gives
// the classical Psi^(n)); in LCDM every raw term has its own growth function (term_g_unmerged).

/**
 * @param sim a CosmoSim with build_lpt(order) done (any grid size: the list depends on order and cosmology only)
 * @returns {{count:number, orders:number[], kind:number[], a:number[], b:number[], c:number[], coefs:Float64Array, eds:boolean}}
 */
export function getUnmergedSpec(sim) {
  const f = sim.term_specs_unmerged();
  const count = f.length / 5;
  const o = { count, orders: [], kind: [], a: [], b: [], c: [], coefs: sim.term_coefs_unmerged(), eds: Math.abs(sim.omega_m() - 1) < 1e-12 };
  for (let t = 0; t < count; t++) { o.orders.push(f[5 * t]); o.kind.push(f[5 * t + 1]); o.a.push(f[5 * t + 2]); o.b.push(f[5 * t + 3]); o.c.push(f[5 * t + 4]); }
  return o;
}

/**
 * g_tau(D) for the unmerged list: closed form in EdS; in LCDM the WASM tables.  In LCDM the sim's own term list IS the raw list
 * (nothing is merged), so term_g(D) - a table lookup - equals term_g_unmerged(D) (which re-integrates the growth ODEs on every call,
 * milliseconds): use the former.  `sim` must have built (at least) the order of `spec`.
 */
export function growthUnmerged(sim, spec, D) {
  if (spec.eds) return spec.orders.map((o, t) => spec.coefs[t] * Math.pow(D, o));
  return Array.from(sim.term_g(D));
}
