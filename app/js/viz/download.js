// download.js - tiny helpers to hand a Blob or a string to the user as a file download.

/** File-name-safe slug: lowercase, non-alphanumerics to '_', trimmed. */
export function slugify(s, fallback = 'plot') {
  const t = String(s ?? '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return t || fallback;
}

/** Download a Blob via a temporary <a download>; the object URL is revoked afterwards. Returns the file name. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.rel = 'noopener'; a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
  return filename;
}

export function downloadText(text, filename, type = 'text/plain') {
  return downloadBlob(new Blob([text], { type: type + ';charset=utf-8' }), filename);
}

/** Flash a short message on a button (e.g. 'copied') and restore its label after `ms`. */
export function flash(btn, text, ms = 1000) {
  if (!btn) return;
  if (btn._flashT) clearTimeout(btn._flashT); else btn._flashLabel = btn.textContent;
  btn.textContent = text;
  btn._flashT = setTimeout(() => { btn.textContent = btn._flashLabel; btn._flashT = 0; }, ms);
}
