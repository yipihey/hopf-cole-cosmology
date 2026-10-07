// colormaps.js - perceptual colormaps and colorbar rendering.
//
// viridis / magma / inferno / twilight / coolwarm / RdBu are 64-entry tables
// sampled from matplotlib (hex-packed rrggbb) with linear interpolation.
// `rdbu` is matplotlib's RdBu reversed (blue = low, red = high) so that it
// reads naturally for over/under-densities. `twilight` is cyclic (phase maps).

import { linearTicks, logTicks, formatLinear, formatLog } from './ticks.js';
import { pinCssSize } from './gpu.js';

const TABLES = {
  viridis: '44015446075a470d6047136548186a481d6f482374482878472d7b46327e453781443b844240864045883e49893d4e8a3a538b38588c365c8d34608d32648e31688e2f6c8e2d708e2c738e2a778e297b8e277f8e26828e24868e238a8d218e8d20928c1f968b1f9a8a1f9e891fa18721a58523a98326ad812ab07f2fb47c35b7793bbb7542be714ac16d52c5695ac86465cb5e6ece5877d15381d34d8bd64695d840a0da39aadc32b5de2bc0df25cae11fd5e21adfe318eae51af4e61efde725',
  magma: '00000402010903031206051a0a08220e0b2b130d34180f3d1d114722115029115a2f116336106b3d0f71440f764a107952137c59157e5f187f651a806b1d81721f817822817e24828426818b2981912b81982d809e2f7fa5317eab337cb2357bba3878c03a76c73d73cd4071d3436ed9466bdf4a68e44f64e95462ed5a5ff1605df4675cf66e5cf8765cfa7d5efb8560fc8e64fd9668fe9d6cfea571feac76feb47bfebb81fec287feca8dfed194fed89afde0a1fde7a9fceeb0fcf6b8fcfdbf',
  inferno: '00000402010a04031207051b0b072410092d150b371b0c41210c4a280b532f0a5b3609613d0965440a684a0c6b510e6c59106e5f136e65156e6c186e721a6e781c6d7f1e6c85216b8c23699225689827669f2a63a52c60ab2f5eb1325ab73557bf3952c43c4eca404acf4446d44842d94d3dde5238e25734e65d2fea632aed6925f06f20f3761bf57d15f78410f98b0bfa9407fb9b06fca309fcaa0ffcb216fbba1ffac228f9c932f7d13df5d949f4e156f2e865f1ef75f3f586f6fa96fcffa4',
  rdbu: '05306109386d0e41791249841752901b5a9c1f63a8246aae2a71b22f79b53480b93a87bd3f8ec04695c4529dc85fa5cd6eaed27bb6d687beda93c6de9dcbe1a7d0e4b1d5e7bbdaeac5dfeccfe4efd5e7f1dbeaf2e1edf3e7f0f4edf2f5f3f5f6f8f4f2f9f0ebf9ebe3fae7dcfbe3d4fcdecdfdd9c4fbd0b9fac8aff8bfa4f7b799f6af8ef4a683f09c7beb9172e6866ae17860dc6e57d7634fd25849cc4c44c6413ec13639bb2a34b61f2eae172aa21328960f278a0b257f082373042167001f',
  twilight: 'e2d9e2ddd9e0d6d7ddcdd3d8c2ced4b6c8cfaac2cc9ebbc993b4c689adc580a5c3779dc27195c06b8dbf6785be647cbc6173ba606ab75f61b45f58b05e4eab5e43a55d3a9e5b309558278b551f7f4f19724915644212573c114b36114232123a3212373711393e113c4713405014445b164865194c701c4e7a20508425508e2c509633509e3b50a54350ab4b50b15452b65d54ba6657be705bc27a61c58468c78f72ca997ccca287ceac94d1b5a1d4beafd8c6bddccdcaded3d4e1d7dce2d9e2',
  coolwarm: '3b4cc03f53c6445acc4961d24e68d8536edd5875e15d7ce66282ea6788ee6c8ff17295f4779af77da0f982a6fb88abfd8fb1fe94b6ff9abbff9fbfffa5c3feaac7fdafcafcb5cdfabad0f8bfd3f6c4d5f3c9d7f0cdd9ecd2dbe8d6dce4dadce0dfdbd9e3d9d3e7d7ceead4c8edd1c2f0cdbbf2cab5f4c6aff5c1a9f6bda2f7b89cf7b396f7ad90f7a889f6a283f59c7df39475f18d6fee8669ec7f63e9785de57058e26952de614dd95847d55042d0473dcb3e38c53334c0282fba162bb40426',
};

/** Decode a hex-packed table into an array of [r,g,b] (0..255). */
function decode(hex) {
  const out = [];
  for (let i = 0; i < hex.length; i += 6) {
    out.push([parseInt(hex.slice(i, i + 2), 16), parseInt(hex.slice(i + 2, i + 4), 16), parseInt(hex.slice(i + 4, i + 6), 16)]);
  }
  return out;
}

/** Build t -> [r,g,b] from a table with linear interpolation. */
function fromTable(hex) {
  const tbl = decode(hex);
  const N = tbl.length;
  return (t) => {
    t = t > 0 ? (t < 1 ? t : 1) : 0; // clamps and maps NaN to 0
    const x = t * (N - 1), i = Math.min(N - 2, Math.floor(x)), f = x - i;
    const a = tbl[i], b = tbl[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
}

/** Each colormap is a function t in [0,1] -> [r,g,b] with channels in 0..255. */
export const COLORMAPS = {
  viridis: fromTable(TABLES.viridis),
  magma: fromTable(TABLES.magma),
  inferno: fromTable(TABLES.inferno),
  rdbu: fromTable(TABLES.rdbu),
  twilight: fromTable(TABLES.twilight),
  gray: (t) => { const v = 255 * (t > 0 ? (t < 1 ? t : 1) : 0); return [v, v, v]; },
  coolwarm: fromTable(TABLES.coolwarm),
};

/** Look up a colormap function by name (falls back to viridis with a warning). */
export function getColormap(name) {
  const f = COLORMAPS[name];
  if (!f) { console.warn(`[viz] unknown colormap "${name}", using viridis`); return COLORMAPS.viridis; }
  return f;
}

/** RGBA lookup table: Uint8Array(size*4), alpha = 255. */
export function cmapLUT(name, size = 256) {
  const f = getColormap(name), lut = new Uint8Array(size * 4);
  for (let i = 0; i < size; i++) {
    const c = f(size > 1 ? i / (size - 1) : 0);
    lut[4 * i] = Math.round(c[0]); lut[4 * i + 1] = Math.round(c[1]);
    lut[4 * i + 2] = Math.round(c[2]); lut[4 * i + 3] = 255;
  }
  return lut;
}

/**
 * Draw a colorbar with tick labels into a plain 2D canvas.
 * @param {HTMLCanvasElement} canvas  orientation defaults to vertical if h >= w, else horizontal
 * @param {string} name   colormap name
 * @param {number} vmin
 * @param {number} vmax
 * @param {{label?:string, log?:boolean, orientation?:'vertical'|'horizontal'}} [opts]
 */
export function renderColorbar(canvas, name, vmin, vmax, { label = '', log = false, orientation } = {}) {
  pinCssSize(canvas);
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth || canvas.width / dpr, ch = canvas.clientHeight || canvas.height / dpr;
  const W = Math.max(1, Math.round(cw * dpr)), H = Math.max(1, Math.round(ch * dpr));
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const vertical = orientation ? orientation === 'vertical' : ch >= cw;
  const color = getComputedStyle(canvas).color || '#888';
  const font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  ctx.font = font; ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1;

  const f = getColormap(name);
  if (log && !(vmin > 0 && vmax > 0)) { console.warn('[viz] colorbar: log scale needs positive range'); log = false; }
  // map value -> fraction along the bar
  const frac = log
    ? (v) => (Math.log10(v) - Math.log10(vmin)) / (Math.log10(vmax) - Math.log10(vmin))
    : (v) => (v - vmin) / (vmax - vmin);
  let ticks, labels;
  if (log) { const lt = logTicks(vmin, vmax); ticks = lt.major; labels = ticks.map(formatLog); }
  else { const lt = linearTicks(vmin, vmax, vertical ? Math.max(2, Math.floor(ch / 40)) : Math.max(2, Math.floor(cw / 70))); ticks = lt.ticks; labels = ticks.map((v) => formatLinear(v, lt.step)); }

  const barThick = 14, pad = 6;
  if (vertical) {
    // bar on the left, labels right, axis label rotated at far right
    const x0 = pad, y0 = pad, h = ch - 2 * pad;
    // gradient drawn as thin rects (device independent)
    for (let y = 0; y < h; y += 1) { const c = f(1 - y / Math.max(1, h - 1)); ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; ctx.fillRect(x0, y0 + y, barThick, 1.2); }
    ctx.fillStyle = color; ctx.strokeRect(x0 + 0.5, y0 + 0.5, barThick - 1, h - 1);
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ticks.forEach((v, k) => {
      const t = frac(v); if (t < -1e-9 || t > 1 + 1e-9) return;
      const y = y0 + (1 - t) * (h - 1);
      ctx.beginPath(); ctx.moveTo(x0 + barThick, y + 0.5); ctx.lineTo(x0 + barThick + 4, y + 0.5); ctx.stroke();
      ctx.fillText(labels[k], x0 + barThick + 7, y);
    });
    if (label) {
      ctx.save(); ctx.translate(cw - 6, y0 + h / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(label, 0, 0); ctx.restore();
    }
  } else {
    const x0 = pad + 4, w = cw - 2 * pad - 8, y0 = pad;
    for (let x = 0; x < w; x += 1) { const c = f(x / Math.max(1, w - 1)); ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; ctx.fillRect(x0 + x, y0, 1.2, barThick); }
    ctx.fillStyle = color; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, barThick - 1);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ticks.forEach((v, k) => {
      const t = frac(v); if (t < -1e-9 || t > 1 + 1e-9) return;
      const x = x0 + t * (w - 1);
      ctx.beginPath(); ctx.moveTo(x + 0.5, y0 + barThick); ctx.lineTo(x + 0.5, y0 + barThick + 4); ctx.stroke();
      ctx.fillText(labels[k], x, y0 + barThick + 6);
    });
    if (label) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(label, x0 + w / 2, ch - 2); }
  }
}
