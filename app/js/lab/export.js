// export.js - PNG export of a field panel (stage canvas + colour bar side by side) and the "copy link" helper.
//
// WebGPU canvases do not preserve their drawing buffer: a drawImage / toBlob taken after the frame has been presented
// can be blank. `redraw` therefore re-renders the view synchronously right before the stage is copied, in the same task
// (Chrome lets drawImage read the not-yet-presented texture of a WebGPU canvas). The composed result lives on an
// ordinary Canvas2D surface, so toBlob is safe.

import { downloadBlob, slugify, flash } from '../viz/download.js';

/** Page background as a solid CSS colour (the colour-bar labels follow the page text colour, so the page background keeps them legible). */
function pageBackground() {
  for (const e of [document.body, document.documentElement]) {
    const c = getComputedStyle(e).backgroundColor;
    const m = c && c.match(/rgba?\(([^)]+)\)/);
    if (m) { const p = m[1].split(',').map(Number); if (p.length < 4 || p[3] > 0.99) return c; }
  }
  return '#fff';
}

/**
 * Compose stage + colour bar at device resolution.
 * @param {{canvas:HTMLCanvasElement, bar?:HTMLCanvasElement, redraw?:()=>void}} o
 * @returns {Promise<Blob>}
 */
export function composePNG({ canvas, bar, redraw }) {
  if (redraw) redraw();                       // synchronous: the GPU texture is still current for drawImage below
  const W = canvas.width, H = canvas.height;
  if (!W || !H) return Promise.reject(new Error('nothing drawn yet'));
  const gap = bar ? Math.round(8 * (window.devicePixelRatio || 1)) : 0;
  const bw = bar ? bar.width : 0;
  const out = document.createElement('canvas');
  out.width = W + gap + bw; out.height = H;
  const ctx = out.getContext('2d');
  ctx.fillStyle = pageBackground();
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(canvas, 0, 0, W, H);
  if (bar && bar.width && bar.height) ctx.drawImage(bar, W + gap, 0, bw, H);
  return new Promise((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('toBlob returned null'))), 'image/png'));
}

/** Name part for a growth factor, e.g. 0.3 -> 'D0.3'. */
export const dTag = (D) => 'D' + String(+Number(D).toPrecision(4));

/** `<kind>_<sub>_D<value>.png`, each part slugified, empty parts dropped. */
export function pngName(kind, sub, D) {
  return [slugify(kind, ''), slugify(sub, ''), dTag(D)].filter(Boolean).join('_') + '.png';
}

/** Compose and download. Flashes the button label. */
export async function exportPNG(opts, filename, btn) {
  try {
    const blob = await composePNG(opts);
    downloadBlob(blob, filename);
    flash(btn, 'saved');
    return blob;
  } catch (e) {
    console.error('[lab] PNG export failed:', e);
    flash(btn, 'failed', 1500);
    return null;
  }
}

/** Small "png" button for a panel header. */
export function pngButton(parent, onClick) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'hcc-btn lab-png'; b.textContent = 'png';
  b.title = 'Download this panel (image and colour bar) as a PNG at device resolution';
  b.addEventListener('click', () => onClick(b));
  parent.appendChild(b);
  return b;
}

/** Copy `text` to the clipboard; without the async Clipboard API (or when it is refused) select it in a temporary input and use execCommand. */
export async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return true; }
  } catch (e) { /* fall through to the selection fallback */ }
  const inp = document.createElement('input');
  inp.value = text; inp.setAttribute('readonly', ''); inp.setAttribute('aria-hidden', 'true');
  inp.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
  document.body.appendChild(inp);
  inp.focus(); inp.select(); inp.setSelectionRange(0, text.length);
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  inp.remove();
  return ok;
}
