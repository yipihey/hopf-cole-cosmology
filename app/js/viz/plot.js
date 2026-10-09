// plot.js - crisp, theme-aware SVG line/scatter plots (no canvas).
//
// Axes, ticks and text use `currentColor`, so a plot inherits the text colour
// of its container and works on light and dark pages. Series use an
// Okabe-Ito categorical palette. The SVG scales with its container (viewBox)
// up to `width` px.

import { COLORMAPS } from './colormaps.js';
import { linearTicks, logTicks, formatLinear, formatLog } from './ticks.js';

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
  }

  /** @param {{xlog?:boolean, ylog?:boolean, xlabel?:string, ylabel?:string, xlim?:number[], ylim?:number[], title?:string}} axes */
  setAxes(axes) { this.axes = { ...this.axes, ...axes }; }

  /**
   * @param {{x:ArrayLike<number>, y:ArrayLike<number>, label?:string, color?:string, dash?:string,
   *          width?:number, points?:boolean, hollow?:boolean, line?:boolean, opacity?:number}[]} series
   */
  setSeries(series) { this.series = series; }

  /** @param {{x:number, label?:string, color?:string}[]} markers vertical marker lines */
  setMarkers(markers) { this.markers = markers; }

  /** Serialized SVG (e.g. for download). */
  toSVGString() { return new XMLSerializer().serializeToString(this.svg); }

  destroy() { this.svg.remove(); }

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
    const A = this.axes, svg = this.svg, W = this.opts.width, H = this.opts.height;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('aria-label', A.title || 'plot');
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
