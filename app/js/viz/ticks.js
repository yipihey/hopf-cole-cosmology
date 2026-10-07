// ticks.js - axis tick generation and number formatting shared by plot.js and
// the colorbar renderer in colormaps.js. Pure functions, no DOM access.

const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };

/** Convert an integer exponent to unicode superscript characters, e.g. -3 -> "⁻³". */
export function superscript(e) {
  return String(e).split('').map((c) => SUP[c] || c).join('');
}

/** Remove floating point noise (0.30000000000000004 -> 0.3). */
function clean(x) { return Number(x.toPrecision(12)); }

/**
 * "Nice" linear ticks (steps of 1, 2 or 5 times a power of ten).
 * @returns {{ticks:number[], step:number}}
 */
export function linearTicks(lo, hi, target = 6) {
  if (!(isFinite(lo) && isFinite(hi))) return { ticks: [], step: 1 };
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi === lo) { const d = Math.abs(lo) || 1; lo -= d / 2; hi += d / 2; }
  const raw = (hi - lo) / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
  const ticks = [];
  const first = Math.ceil(lo / step - 1e-9);
  for (let k = first; k * step <= hi + step * 1e-9; k++) ticks.push(clean(k * step));
  return { ticks, step };
}

/**
 * Logarithmic ticks for a positive range.
 * major: labelled ticks; minor: unlabelled (2..9 x 10^k).
 * When the range spans fewer than ~2 decades the 1-2-5 values are labelled instead.
 */
export function logTicks(lo, hi) {
  if (!(lo > 0 && hi > 0 && isFinite(lo) && isFinite(hi))) return { major: [], minor: [] };
  if (hi < lo) [lo, hi] = [hi, lo];
  const e0 = Math.floor(Math.log10(lo) - 1e-9), e1 = Math.ceil(Math.log10(hi) + 1e-9);
  const decades = Math.log10(hi / lo);
  const inR = (v) => v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9);
  const major = [], minor = [];
  if (decades < 2) {
    for (let e = e0; e <= e1; e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (inR(v)) major.push(v); }
    for (let e = e0; e <= e1; e++) for (const m of [3, 4, 6, 7, 8, 9]) { const v = m * Math.pow(10, e); if (inR(v)) minor.push(v); }
  } else {
    const stride = Math.max(1, Math.ceil(decades / 7));
    for (let e = e0; e <= e1; e++) {
      const v = Math.pow(10, e);
      if (inR(v) && (e - e0) % stride === 0) major.push(v);
      if (stride === 1) for (let m = 2; m <= 9; m++) { const w = m * v; if (inR(w)) minor.push(w); }
    }
  }
  return { major, minor };
}

/** Format a linear tick value given the tick step (decides decimals / exponent form). */
export function formatLinear(v, step = 1) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-3) return formatExp(v);
  const dec = Math.min(8, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(dec);
}

/** Compact exponent form: 2.5e5, 1e-4. */
export function formatExp(v) {
  if (v === 0) return '0';
  const s = v.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e');
  return s;
}

/** Label for a log-axis tick: 10^n for exact powers of ten, otherwise m x 10^n. */
export function formatLog(v) {
  const e = Math.floor(Math.log10(v) + 1e-9);
  const m = Math.round(v / Math.pow(10, e));
  if (m === 1) return '10' + superscript(e);
  if (m === 10) return '10' + superscript(e + 1);
  return m + '×10' + superscript(e);
}
