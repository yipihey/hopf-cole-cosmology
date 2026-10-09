// plot.js - crisp, theme-aware SVG line/scatter plots (no canvas).
//
// Axes, ticks and text use `currentColor`, so a plot inherits the text colour
// of its container and works on light and dark pages. Series use an
// Okabe-Ito categorical palette. The SVG scales with its container (viewBox)
// up to `width` px.

import { COLORMAPS } from './colormaps.js';
import { linearTicks, logTicks, formatLinear, formatLog } from './ticks.js';
import { downloadText, slugify, flash } from './download.js';

const NS = 'http://www.w3.org/2000/svg';

/** Okabe-Ito colour-blind-friendly palette (black replaced by mid grey for dark pages). */
export const PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#999999'];

let uid = 0;

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

const fmt = (x) => (Math.round(x * 100) / 100).toString();

export class LinePlot {
  /**
   * @param {HTMLElement} container
   * @param {{width?:number, height?:number, margin?:{l?:number,r?:number,t?:number,b?:number}}} [opts]
   */
  constructor(container, opts = {}) {
    this.container = container;
    this.opts = { width: 520, height: 340, ...opts };
    this.userMargin = opts.margin || {};
    this.axes = { xlog: false, ylog: false };
    this.series = [];
    this.markers = [];
    this.id = ++uid;
    const { width: W, height: H } = this.opts;
    this.svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`, role: 'img', class: 'hcc-plot',
      style: `display:block;width:100%;height:auto;max-width:${W}px;overflow:visible;color:inherit;font:12px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif`,
    });
    container.appendChild(this.svg);
    // interactive view (overrides the caller's axes until reset): wheel = zoom, drag = pan, double-click = reset,
    // toolbar = lin/log toggles and editable ranges
    this.view = { xlim: null, ylim: null, xlog: null, ylog: null };
    this._lastDefaults = null;
    this.interactive = opts.interactive !== false;
    if (this.interactive) this._buildTools();
  }

  /** @param {{xlog?:boolean, ylog?:boolean, xlabel?:string, ylabel?:string, xlim?:number[], ylim?:number[], title?:string}} axes */
  setAxes(axes) { this.axes = { ...this.axes, ...axes }; }

  /** Axes with the interactive overrides applied; a change of the caller's lin/log default drops the override of that axis. */
  _effAxes() {
    const A = this.axes, v = this.view;
    const d = { xlog: !!A.xlog, ylog: !!A.ylog };
    if (this._lastDefaults) {
      if (this._lastDefaults.xlog !== d.xlog) { v.xlog = null; v.xlim = null; }
      if (this._lastDefaults.ylog !== d.ylog) { v.ylog = null; v.ylim = null; }
    }
    this._lastDefaults = d;
    return { ...A, xlog: v.xlog ?? d.xlog, ylog: v.ylog ?? d.ylog, xlim: v.xlim ?? A.xlim, ylim: v.ylim ?? A.ylim };
  }

  /** Forget zoom, pan and lin/log overrides. */
  resetView() { this.view = { xlim: null, ylim: null, xlog: null, ylog: null }; this.draw(); }

  // --- interaction ----------------------------------------------------------

  _buildTools() {
    const doc = document;
    const bar = this.tools = doc.createElement('div');
    bar.className = 'hcc-plot-tools';
    bar.style.maxWidth = `${this.opts.width}px`;
    bar.title = 'Wheel over the plot: zoom (over an axis: that axis only). Drag: pan. Double-click: reset.';
    const mk = (tag, cls, parent, text) => { const e = doc.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; parent.appendChild(e); return e; };
    const axisTools = (key, label) => {
      const g = mk('span', 'hcc-pt-axis', bar);
      mk('span', 'hcc-pt-label', g, label);
      const b = mk('button', 'hcc-pt-btn', g, 'lin'); b.type = 'button'; b.title = `${label} axis: linear / logarithmic`;
      b.addEventListener('click', () => this._toggleLog(key));
      const lo = mk('input', 'hcc-pt-in', g); lo.type = 'text'; lo.inputMode = 'decimal'; lo.title = `${label} minimum`;
      mk('span', 'hcc-pt-dash', g, '–');
      const hi = mk('input', 'hcc-pt-in', g); hi.type = 'text'; hi.inputMode = 'decimal'; hi.title = `${label} maximum`;
      const apply = () => this._setRange(key, parseFloat(lo.value), parseFloat(hi.value));
      for (const inp of [lo, hi]) { inp.addEventListener('change', apply); inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { apply(); inp.blur(); } }); }
      return { b, lo, hi };
    };
    this._tx = axisTools('x', 'x');
    this._ty = axisTools('y', 'y');
    const r = mk('button', 'hcc-pt-btn hcc-pt-reset', bar, 'reset'); r.type = 'button'; r.title = 'Reset zoom, ranges and scales';
    r.addEventListener('click', () => this.resetView());
    const ex = mk('span', 'hcc-pt-export', bar);
    const bs = mk('button', 'hcc-pt-btn', ex, 'svg'); bs.type = 'button'; bs.title = 'Download this plot as a standalone SVG file';
    bs.addEventListener('click', () => { this.downloadSVG(); flash(bs, 'saved'); });
    const bc = mk('button', 'hcc-pt-btn', ex, 'csv'); bc.type = 'button'; bc.title = 'Download the plotted series as a CSV file (full precision)';
    bc.addEventListener('click', () => { this.downloadCSV(); flash(bc, 'saved'); });
    this._readout = mk('span', 'hcc-pt-readout', bar, '');
    this.container.appendChild(bar);
    this._bindEvents();
  }

  _syncTools() {
    if (!this.tools || !this._scales) return;
    const A = this._effAxes(), S = this._scales;
    const f = (v) => Number.isFinite(v) ? String(+v.toPrecision(4)) : '';
    this._tx.b.textContent = A.xlog ? 'log' : 'lin'; this._ty.b.textContent = A.ylog ? 'log' : 'lin';
    if (document.activeElement !== this._tx.lo) this._tx.lo.value = f(S.x0);
    if (document.activeElement !== this._tx.hi) this._tx.hi.value = f(S.x1);
    if (document.activeElement !== this._ty.lo) this._ty.lo.value = f(S.y0);
    if (document.activeElement !== this._ty.hi) this._ty.hi.value = f(S.y1);
    const changed = this.view.xlim || this.view.ylim || this.view.xlog !== null || this.view.ylog !== null;
    this.tools.classList.toggle('is-changed', !!changed);
  }

  _toggleLog(key) {
    const A = this._effAxes();
    const cur = key === 'x' ? A.xlog : A.ylog;
    this.view[key + 'log'] = !cur;
    const lim = this.view[key + 'lim'];
    if (!cur && lim && lim[0] <= 0) this.view[key + 'lim'] = null;   // switching to log with a non-positive range: back to auto
    this.draw();
  }

  _setRange(key, lo, hi) {
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) { this._syncTools(); return; }
    const A = this._effAxes();
    if ((key === 'x' ? A.xlog : A.ylog) && lo <= 0) { this._syncTools(); return; }
    this.view[key + 'lim'] = [lo, hi];
    this.draw();
  }

  /** Pointer position in viewBox units. */
  _pos(ev) {
    const r = this.svg.getBoundingClientRect();
    const k = this.opts.width / Math.max(1, r.width);
    return { px: (ev.clientX - r.left) * k, py: (ev.clientY - r.top) * k };
  }

  /** Which region the pointer is in: 'plot', 'x' (x-axis strip), 'y' (y-axis strip) or null. */
  _region(px, py) {
    const S = this._scales; if (!S) return null;
    const { m, pw, ph } = S;
    const inX = px >= m.l && px <= m.l + pw, inY = py >= m.t && py <= m.t + ph;
    if (inX && inY) return 'plot';
    if (inX && py > m.t + ph && py <= this.opts.height) return 'x';
    if (inY && px < m.l && px >= 0) return 'y';
    return null;
  }

  /** Transform to/from axis space (log10 for log axes). */
  _t(key) {
    const A = this._effAxes(), log = key === 'x' ? A.xlog : A.ylog;
    return log ? { f: Math.log10, g: (u) => Math.pow(10, u) } : { f: (v) => v, g: (u) => u };
  }

  /** Current limits of an axis in axis space. */
  _lim(key) { const S = this._scales, t = this._t(key); return key === 'x' ? [t.f(S.x0), t.f(S.x1)] : [t.f(S.y0), t.f(S.y1)]; }

  _zoomAxis(key, center, factor) {
    const t = this._t(key), [a, b] = this._lim(key);
    const c = t.f(center);
    const na = c - (c - a) / factor, nb = c + (b - c) / factor;
    if (!(nb > na) || !Number.isFinite(na) || !Number.isFinite(nb)) return;
    if (Math.abs(nb - na) < 1e-9 * (Math.abs(c) + 1e-300)) return;
    this.view[key + 'lim'] = [t.g(na), t.g(nb)];
  }

  _panAxis(key, frac) {
    const t = this._t(key), [a, b] = this._lim(key);
    const d = (b - a) * frac;
    this.view[key + 'lim'] = [t.g(a + d), t.g(b + d)];
  }

  _bindEvents() {
    const svg = this.svg;
    svg.style.touchAction = 'pan-y';
    svg.addEventListener('wheel', (ev) => {
      const { px, py } = this._pos(ev);
      const reg = this._region(px, py);
      if (!reg || !this._scales) return;
      ev.preventDefault();
      const S = this._scales;
      const factor = Math.exp(-Math.sign(ev.deltaY) * Math.min(1, Math.abs(ev.deltaY) / 100) * 0.25);
      const xc = S.x0 === undefined ? 0 : this._t('x').g(this._lim('x')[0] + (px - S.m.l) / S.pw * (this._lim('x')[1] - this._lim('x')[0]));
      const yc = this._t('y').g(this._lim('y')[0] + (S.m.t + S.ph - py) / S.ph * (this._lim('y')[1] - this._lim('y')[0]));
      if (reg === 'plot' || reg === 'x') this._zoomAxis('x', xc, factor);
      if (reg === 'plot' || reg === 'y') this._zoomAxis('y', yc, factor);
      this.draw();
    }, { passive: false });
    let drag = null;
    svg.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0) return;
      const { px, py } = this._pos(ev);
      const reg = this._region(px, py);
      if (!reg) return;
      drag = { reg, px, py, moved: false };
      svg.setPointerCapture(ev.pointerId);
    });
    svg.addEventListener('pointermove', (ev) => {
      const { px, py } = this._pos(ev);
      if (drag) {
        const S = this._scales;
        const dx = px - drag.px, dy = py - drag.py;
        if (Math.abs(dx) + Math.abs(dy) < 1) return;
        drag.moved = true;
        if (drag.reg === 'plot' || drag.reg === 'x') this._panAxis('x', -dx / S.pw);
        if (drag.reg === 'plot' || drag.reg === 'y') this._panAxis('y', dy / S.ph);
        drag.px = px; drag.py = py;
        svg.style.cursor = 'grabbing';
        this.draw();
        return;
      }
      const reg = this._region(px, py);
      svg.style.cursor = reg === 'plot' ? 'crosshair' : reg ? 'ew-resize' : '';
      if (reg === 'y') svg.style.cursor = 'ns-resize';
      if (this._readout) {
        if (reg === 'plot') {
          const S = this._scales;
          const x = this._t('x').g(this._lim('x')[0] + (px - S.m.l) / S.pw * (this._lim('x')[1] - this._lim('x')[0]));
          const y = this._t('y').g(this._lim('y')[0] + (S.m.t + S.ph - py) / S.ph * (this._lim('y')[1] - this._lim('y')[0]));
          this._readout.textContent = `x = ${+x.toPrecision(4)}, y = ${+y.toPrecision(4)}`;
        } else this._readout.textContent = '';
      }
    });
    const end = (ev) => { if (!drag) return; drag = null; svg.style.cursor = ''; try { svg.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ } };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
    svg.addEventListener('pointerleave', () => { if (!drag && this._readout) this._readout.textContent = ''; });
    svg.addEventListener('dblclick', (ev) => { ev.preventDefault(); this.resetView(); });
  }

  /**
   * @param {{x:ArrayLike<number>, y:ArrayLike<number>, label?:string, color?:string, dash?:string,
   *          width?:number, points?:boolean, hollow?:boolean, line?:boolean, opacity?:number}[]} series
   */
  setSeries(series) { this.series = series; }

  /** @param {{x:number, label?:string, color?:string}[]} markers vertical marker lines */
  setMarkers(markers) { this.markers = markers; }

  /** Serialized SVG (e.g. for download). */
  toSVGString() { return new XMLSerializer().serializeToString(this.svg); }

  /** Self-contained SVG document: xmlns, explicit size, `color: #222` (so currentColor renders outside the page) and a white background. */
  toStandaloneSVG() {
    const W = this.opts.width, H = this.opts.height;
    const c = this.svg.cloneNode(true);
    c.setAttribute('xmlns', NS);
    c.setAttribute('width', W); c.setAttribute('height', H);
    c.setAttribute('style', 'color:#222;overflow:visible;font:12px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif');
    const bg = document.createElementNS(NS, 'rect');
    for (const [k, v] of Object.entries({ x: 0, y: 0, width: W, height: H, fill: '#fff' })) bg.setAttribute(k, v);
    c.insertBefore(bg, c.firstChild);
    let out = new XMLSerializer().serializeToString(c);
    if (!/^<svg[^>]*\sxmlns=/.test(out)) out = out.replace(/^<svg/, `<svg xmlns="${NS}"`);
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + out + '\n';
  }

  /** File stem from the title (or the y label), slugified. */
  _fileStem() { return slugify(this.axes.title || this.axes.ylabel || 'plot'); }

  downloadSVG() { return downloadText(this.toStandaloneSVG(), this._fileStem() + '.svg', 'image/svg+xml'); }

  /**
   * The plotted series as CSV. When all series share the same x array: wide format `x,<label1>,<label2>,...`;
   * otherwise long format `series,x,y` with one row per point. Numbers are written with full (shortest round-trip) precision.
   */
  toCSV() {
    const S = this.series.filter((s) => s && s.x && s.y);
    const name = (s, k) => (s.label ? String(s.label) : `series${k + 1}`);
    const q = (t) => (/[",\n\r]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t);
    const num = (v) => String(v);
    const n0 = S.length ? Math.min(S[0].x.length, S[0].y.length) : 0;
    const shared = S.length > 0 && S.every((s) => {
      if (Math.min(s.x.length, s.y.length) !== n0 || s.x.length !== S[0].x.length) return false;
      for (let i = 0; i < n0; i++) if (s.x[i] !== S[0].x[i] && !(s.x[i] !== s.x[i] && S[0].x[i] !== S[0].x[i])) return false;
      return true;
    });
    const L = [];
    if (shared) {
      L.push(['x', ...S.map(name)].map(q).join(','));
      for (let i = 0; i < n0; i++) L.push([num(S[0].x[i]), ...S.map((s) => num(s.y[i]))].join(','));
    } else {
      L.push('series,x,y');
      S.forEach((s, k) => { const nm = q(name(s, k)), n = Math.min(s.x.length, s.y.length); for (let i = 0; i < n; i++) L.push(`${nm},${num(s.x[i])},${num(s.y[i])}`); });
    }
    return L.join('\n') + '\n';
  }

  downloadCSV() { return downloadText(this.toCSV(), this._fileStem() + '.csv', 'text/csv'); }

  destroy() { this.svg.remove(); if (this.tools) this.tools.remove(); }

  // --- hooks overridden by WaterfallPlot ------------------------------------
  _drawSeries() { return this.series; }
  _showYTicks() { return true; }
  _defaultMargin() { return {}; }
  _decorate() {}

  // --- helpers ------------------------------------------------------------

  _dataRange(series, key, log) {
    let lo = Infinity, hi = -Infinity;
    for (const s of series) {
      const a = s[key];
      for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (!Number.isFinite(v) || (log && v <= 0)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    return [lo, hi];
  }

  _limits(series, key, log, user, pad) {
    let [lo, hi] = this._dataRange(series, key, log);
    if (!(lo <= hi)) { lo = log ? 1 : 0; hi = log ? 10 : 1; }
    if (lo === hi) { if (log) { lo /= 2; hi *= 2; } else { const d = Math.abs(lo) || 1; lo -= d / 2; hi += d / 2; } }
    if (log) { const e = Math.pow(hi / lo, pad); lo /= e; hi *= e; }
    else { const d = (hi - lo) * pad; lo -= d; hi += d; }
    if (user && user.length === 2) {
      if (Number.isFinite(user[0]) && !(log && user[0] <= 0)) lo = user[0];
      if (Number.isFinite(user[1]) && !(log && user[1] <= 0)) hi = user[1];
    }
    return [lo, hi];
  }

  /** (Re)render the SVG. */
  draw() {
    const A = this._effAxes(), svg = this.svg, W = this.opts.width, H = this.opts.height;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('aria-label', A.title || (A.ylabel && A.xlabel ? `${A.ylabel} versus ${A.xlabel}` : 'plot'));
    const series = this._drawSeries();
    const [x0, x1] = this._limits(series, 'x', A.xlog, A.xlim, 0);
    const [y0, y1] = this._limits(series, 'y', A.ylog, A.ylim, A.ylog ? 0.03 : 0.05);

    // tick sets (needed first so the left margin can fit the labels)
    const xt = A.xlog ? this._logTicks(x0, x1) : this._linTicks(x0, x1, Math.max(3, Math.floor(this.opts.width / 90)));
    const yt = A.ylog ? this._logTicks(y0, y1) : this._linTicks(y0, y1, Math.max(3, Math.floor(this.opts.height / 55)));
    const maxYLabel = this._showYTicks() ? Math.max(0, ...yt.labels.map((s) => s.length)) : 0;
    const dm = this._defaultMargin();
    const m = {
      l: this.userMargin.l ?? (dm.l ?? (14 * (A.ylabel ? 1 : 0) + 14 + maxYLabel * 6.6 + 6)),
      r: this.userMargin.r ?? (dm.r ?? 16),
      t: this.userMargin.t ?? (A.title ? 30 : 12),
      b: this.userMargin.b ?? (A.xlabel ? 46 : 30),
    };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const lx = A.xlog ? [Math.log10(x0), Math.log10(x1)] : [x0, x1];
    const ly = A.ylog ? [Math.log10(y0), Math.log10(y1)] : [y0, y1];
    const sx = (v) => m.l + ((A.xlog ? Math.log10(v) : v) - lx[0]) / (lx[1] - lx[0]) * pw;
    const sy = (v) => m.t + ph - ((A.ylog ? Math.log10(v) : v) - ly[0]) / (ly[1] - ly[0]) * ph;
    this._scales = { sx, sy, m, pw, ph, x0, x1, y0, y1 };

    const clipId = `hcc-clip-${this.id}`;
    const defs = el('defs', null, svg);
    el('rect', { x: m.l, y: m.t, width: pw, height: ph }, el('clipPath', { id: clipId }, defs));

    // grid + ticks
    const grid = el('g', { stroke: 'currentColor', 'stroke-opacity': 0.12, 'stroke-width': 1, fill: 'none' }, svg);
    const ticks = el('g', { stroke: 'currentColor', 'stroke-width': 1 }, svg);
    const labels = el('g', { fill: 'currentColor' }, svg);
    xt.major.forEach((v, k) => {
      const x = Math.round(sx(v)) + 0.5;
      if (x < m.l - 0.5 || x > m.l + pw + 0.5) return;
      el('line', { x1: x, x2: x, y1: m.t, y2: m.t + ph }, grid);
      el('line', { x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 5 }, ticks);
      el('text', { x, y: m.t + ph + 18, 'text-anchor': 'middle', stroke: 'none' }, labels, xt.labels[k]);
    });
    xt.minor.forEach((v) => { const x = Math.round(sx(v)) + 0.5; if (x >= m.l && x <= m.l + pw) el('line', { x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 3, 'stroke-opacity': 0.6 }, ticks); });
    if (this._showYTicks()) {
      yt.major.forEach((v, k) => {
        const y = Math.round(sy(v)) + 0.5;
        if (y < m.t - 0.5 || y > m.t + ph + 0.5) return;
        el('line', { x1: m.l, x2: m.l + pw, y1: y, y2: y }, grid);
        el('line', { x1: m.l - 5, x2: m.l, y1: y, y2: y }, ticks);
        el('text', { x: m.l - 8, y: y + 4, 'text-anchor': 'end', stroke: 'none' }, labels, yt.labels[k]);
      });
      yt.minor.forEach((v) => { const y = Math.round(sy(v)) + 0.5; if (y >= m.t && y <= m.t + ph) el('line', { x1: m.l - 3, x2: m.l, y1: y, y2: y, 'stroke-opacity': 0.6 }, ticks); });
    }

    // frame
    el('rect', { x: m.l + 0.5, y: m.t + 0.5, width: pw, height: ph, fill: 'none', stroke: 'currentColor', 'stroke-width': 1 }, svg);

    // series
    const plot = el('g', { 'clip-path': `url(#${clipId})` }, svg);
    this._decorate(plot, { sx, sy, m, pw, ph });
    series.forEach((s, k) => {
      const color = s.color || PALETTE[k % PALETTE.length];
      const op = s.opacity ?? 1;
      const n = Math.min(s.x.length, s.y.length);
      const ok = (i) => {
        const x = s.x[i], y = s.y[i];
        return Number.isFinite(x) && Number.isFinite(y) && !(A.xlog && x <= 0) && !(A.ylog && y <= 0);
      };
      if (s.line !== false) {
        let d = '', pen = false;
        for (let i = 0; i < n; i++) {
          if (!ok(i)) { pen = false; continue; }
          d += (pen ? 'L' : 'M') + fmt(sx(s.x[i])) + ' ' + fmt(sy(s.y[i]));
          pen = true;
        }
        if (d) el('path', { d, fill: 'none', stroke: color, 'stroke-width': s.width ?? 1.5, 'stroke-dasharray': s.dash, 'stroke-opacity': op, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, plot);
      }
      if (s.points) {
        const g = el('g', s.hollow ? { fill: 'none', stroke: color, 'stroke-width': 1.5, 'stroke-opacity': op } : { fill: color, 'fill-opacity': op }, plot);
        for (let i = 0; i < n; i++) if (ok(i)) el('circle', { cx: fmt(sx(s.x[i])), cy: fmt(sy(s.y[i])), r: s.radius ?? 2.5 }, g);
      }
    });

    // vertical markers
    this.markers.forEach((mk, k) => {
      if (!Number.isFinite(mk.x) || (A.xlog && mk.x <= 0)) return;
      const x = sx(mk.x);
      if (x < m.l || x > m.l + pw) return;
      const color = mk.color || 'currentColor';
      el('line', { x1: x, x2: x, y1: m.t, y2: m.t + ph, stroke: color, 'stroke-dasharray': '5 3', 'stroke-width': 1.2, 'stroke-opacity': 0.8 }, svg);
      if (mk.label) {
        const t = el('text', { x: x + 4, y: m.t + 12 + 12 * (k % 2), fill: color, 'font-size': 11 }, svg, mk.label);
        if (x > m.l + pw - 80) { t.setAttribute('x', x - 4); t.setAttribute('text-anchor', 'end'); }
      }
    });

    // titles / axis labels
    if (A.title) el('text', { x: m.l + pw / 2, y: 18, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600, fill: 'currentColor' }, svg, A.title);
    if (A.xlabel) el('text', { x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle', fill: 'currentColor' }, svg, A.xlabel);
    if (A.ylabel) el('text', { transform: `translate(14 ${m.t + ph / 2}) rotate(-90)`, 'text-anchor': 'middle', fill: 'currentColor' }, svg, A.ylabel);

    this._legend(series, m, pw);
    this._syncTools();
  }

  _legend(series, m, pw) {
    const entries = series.map((s, k) => ({ s, k })).filter((e) => e.s.label);
    if (!entries.length) return;
    const w = Math.max(...entries.map((e) => e.s.label.length)) * 6.6 + 34;
    const g = el('g', { 'font-size': 12 }, this.svg);
    const x1 = m.l + pw - 8, x0 = x1 - w;
    entries.forEach((e, row) => {
      const y = m.t + 14 + row * 16, color = e.s.color || PALETTE[e.k % PALETTE.length];
      if (e.s.line !== false) el('line', { x1: x0, x2: x0 + 20, y1: y - 4, y2: y - 4, stroke: color, 'stroke-width': e.s.width ?? 1.5, 'stroke-dasharray': e.s.dash, 'stroke-opacity': e.s.opacity ?? 1 }, g);
      if (e.s.points) el('circle', e.s.hollow ? { cx: x0 + 10, cy: y - 4, r: 2.5, fill: 'none', stroke: color, 'stroke-width': 1.5 } : { cx: x0 + 10, cy: y - 4, r: 2.5, fill: color }, g);
      el('text', { x: x0 + 26, y, fill: 'currentColor' }, g, e.s.label);
    });
  }

  _linTicks(lo, hi, target) {
    const t = linearTicks(lo, hi, target);
    return { major: t.ticks, minor: [], labels: t.ticks.map((v) => formatLinear(v, t.step)) };
  }

  _logTicks(lo, hi) {
    const t = logTicks(lo, hi);
    return { major: t.major, minor: t.minor, labels: t.major.map(formatLog) };
  }
}

/**
 * Stack of 1D curves offset vertically (e.g. u(x) at successive times).
 * Same API as LinePlot; series[k] is shifted up by k * offset and labelled at
 * its right end. Curve colours come from `colormap` unless a series sets `color`.
 */
export class WaterfallPlot extends LinePlot {
  /**
   * @param {HTMLElement} container
   * @param {{width?:number,height?:number,margin?:object,offset?:number,colormap?:string}} [opts]
   *   offset: vertical spacing between curves in y units (default: 0.6 x max peak-to-peak)
   */
  constructor(container, opts = {}) {
    super(container, opts);
    this.offset = opts.offset;
    this.colormap = opts.colormap || 'viridis';
  }

  _defaultMargin() { return { l: this.axes.ylabel ? 28 : 12, r: 60 }; }
  _showYTicks() { return false; }

  _amplitude() {
    let amp = 0;
    for (const s of this.series) {
      const [lo, hi] = this._dataRange([s], 'y', false);
      if (hi - lo > amp) amp = hi - lo;
    }
    return amp || 1;
  }

  _drawSeries() {
    const K = this.series.length;
    const off = this.offset ?? 0.6 * this._amplitude();
    this._off = off;
    const cm = COLORMAPS[this.colormap] || COLORMAPS.viridis;
    return this.series.map((s, k) => {
      const c = cm(K > 1 ? 0.15 + 0.7 * k / (K - 1) : 0.5);
      const y = new Float64Array(s.y.length);
      for (let i = 0; i < y.length; i++) y[i] = s.y[i] + k * off;
      return { ...s, y, base: k * off, label: undefined, _label: s.label, color: s.color || `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})` };
    });
  }

  _decorate(plot, { sx, sy, m, pw }) {
    // faint baseline per curve (drawn under the curves)
    const series = this._drawSeries();
    series.forEach((s) => {
      const y = sy(s.base);
      el('line', { x1: m.l, x2: m.l + pw, y1: y, y2: y, stroke: 'currentColor', 'stroke-opacity': 0.12, 'stroke-dasharray': '2 3' }, plot);
    });
    // labels at the right edge, outside the clip region
    const g = el('g', { fill: 'currentColor', 'font-size': 11 }, this.svg);
    series.forEach((s) => {
      if (s._label) el('text', { x: m.l + pw + 6, y: sy(s.base) + 4, fill: s.color }, g, s._label);
    });
  }
}
