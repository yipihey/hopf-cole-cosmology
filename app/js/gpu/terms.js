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
