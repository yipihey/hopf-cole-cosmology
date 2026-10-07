// dom.js - small DOM helpers shared by the lab modules (on top of viz/ui.js styles).

export function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

/** Labelled <select> whose options can be replaced. options: [[value,label]]. */
export function sel(parent, { label = '', options = [], value, onChange, title } = {}) {
  const wrap = el('label', 'hcc-ctl hcc-select', parent);
  if (title) wrap.title = title;
  if (label) el('span', 'hcc-label', wrap, label);
  const s = el('select', 'form-select form-select-sm', wrap);
  const fill = (opts, v) => {
    s.textContent = '';
    for (const [val, text] of opts) { const o = el('option', null, s, text); o.value = String(val); }
    if (v !== undefined) s.value = String(v);
  };
  fill(options, value);
  s.addEventListener('change', () => onChange && onChange(s.value));
  return { el: wrap, select: s, get: () => s.value, set(v) { s.value = String(v); }, setOptions: fill, show(b) { wrap.hidden = !b; } };
}

/** Labelled <input type=number>. */
export function numIn(parent, { label = '', value = 0, min, max, step = 1, onChange, width = '4.2rem', title } = {}) {
  const wrap = el('label', 'hcc-ctl lab-num', parent);
  if (title) wrap.title = title;
  if (label) el('span', 'hcc-label', wrap, label);
  const i = el('input', 'form-control form-control-sm', wrap);
  i.type = 'number'; i.value = value; i.step = step;
  if (min !== undefined) i.min = min;
  if (max !== undefined) i.max = max;
  i.style.width = width;
  i.addEventListener('change', () => {
    let v = Number(i.value);
    if (!Number.isFinite(v)) v = value;
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    i.value = v;
    if (onChange) onChange(v);
  });
  return { el: wrap, input: i, get: () => Number(i.value), set(v) { i.value = v; }, show(b) { wrap.hidden = !b; } };
}

export const tick = () => new Promise((r) => setTimeout(r, 0));
/** Resolve after the next paint (with a timeout fallback for background tabs). */
export const paint = () => new Promise((r) => {
  let done = false;
  const f = () => { if (!done) { done = true; r(); } };
  requestAnimationFrame(f);
  setTimeout(f, 60);
});

export function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms'; }
export function fmtNum(v, d = 3) {
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '–';
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(2);
  return String(Number(v.toPrecision(d)));
}
