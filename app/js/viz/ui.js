// ui.js - tiny framework-free helpers for labelled controls.
// All helpers append to `parent` and return a small handle. Styling lives in
// ../../css/hcc.css (classes .hcc-ctl, .hcc-label, .hcc-row ...).

function make(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

/** Default number formatting for slider read-outs. */
function defaultFormat(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  return String(Number(v.toPrecision(3)));
}

/** Horizontal flex row container. */
export function row(parent, cls) {
  return make('div', 'hcc-row' + (cls ? ' ' + cls : ''), parent);
}

/**
 * Labelled range slider with value read-out. With log=true, min/max must be > 0
 * and the slider is linear in log(value).
 * @returns {{el:HTMLElement, get:()=>number, set:(v:number)=>void}}
 */
export function slider(parent, { label, min = 0, max = 1, step, value, log = false, format = defaultFormat, onInput } = {}) {
  const wrap = make('label', 'hcc-ctl hcc-slider', parent);
  make('span', 'hcc-label', wrap, label || '');
  const input = make('input', null, wrap);
  input.type = 'range';
  const out = make('output', 'hcc-value', wrap);
  const RES = 1000;
  let lo = min, hi = max;
  if (log) { if (!(min > 0 && max > 0)) throw new Error('slider: log scale needs min,max > 0'); lo = Math.log(min); hi = Math.log(max); input.min = 0; input.max = RES; input.step = 1; }
  else { input.min = min; input.max = max; input.step = step === undefined ? (max - min) / 200 : step; }
  const toValue = () => (log ? Math.exp(lo + (hi - lo) * Number(input.value) / RES) : Number(input.value));
  const fromValue = (v) => (log ? String(Math.round((Math.log(v) - lo) / (hi - lo) * RES)) : String(v));
  const show = () => { out.textContent = format(toValue()); };
  input.value = fromValue(value === undefined ? min : value);
  show();
  input.addEventListener('input', () => { show(); if (onInput) onInput(toValue(), input); });
  return {
    el: wrap,
    get: toValue,
    set(v) { input.value = fromValue(v); show(); },
  };
}

/** Labelled <select>. options: [{value, label}] (value may be any string/number). */
export function select(parent, { label, options = [], value, onChange } = {}) {
  const wrap = make('label', 'hcc-ctl hcc-select', parent);
  make('span', 'hcc-label', wrap, label || '');
  const sel = make('select', 'form-select form-select-sm', wrap);
  options.forEach((o) => { const opt = make('option', null, sel, o.label ?? String(o.value)); opt.value = o.value; });
  if (value !== undefined) sel.value = value;
  sel.addEventListener('change', () => { if (onChange) onChange(sel.value, sel); });
  return { el: wrap, get: () => sel.value, set(v) { sel.value = v; } };
}

/** Labelled checkbox. */
export function checkbox(parent, { label, value = false, onChange } = {}) {
  const wrap = make('label', 'hcc-ctl hcc-check', parent);
  const cb = make('input', null, wrap);
  cb.type = 'checkbox'; cb.checked = !!value;
  make('span', 'hcc-label', wrap, label || '');
  cb.addEventListener('change', () => { if (onChange) onChange(cb.checked, cb); });
  return { el: wrap, get: () => cb.checked, set(v) { cb.checked = !!v; } };
}

/** Push button. */
export function button(parent, { label, onClick } = {}) {
  const b = make('button', 'hcc-btn btn btn-sm btn-outline-secondary', parent, label || '');
  b.type = 'button';
  b.addEventListener('click', (ev) => { if (onClick) onClick(ev, b); });
  return { el: b, setLabel(t) { b.textContent = t; }, setDisabled(d) { b.disabled = !!d; } };
}

/** Label + monospace text read-out. */
export function readout(parent, { label } = {}) {
  const wrap = make('div', 'hcc-ctl hcc-readout', parent);
  if (label) make('span', 'hcc-label', wrap, label);
  const out = make('output', 'hcc-value', wrap);
  return { el: wrap, set(text) { out.textContent = text; } };
}
