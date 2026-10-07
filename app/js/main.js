// Entry point of the Interactive Laboratory.
//
// The modules are imported dynamically so that a failure while loading any of them (a syntax error, a WebGPU global used at
// module top level in a browser without WebGPU, ...) shows up in the status line instead of leaving it at "starting …".
// The lab builds its controls first, then loads the WASM core and probes the GPU (see Lab.start).

const fatal = (e) => {
  console.error(e);
  const s = document.getElementById('lab-status-text');
  if (s) { s.textContent = 'Failed to start: ' + (e && e.message ? e.message : e); s.className = 'err'; }
  const d = document.getElementById('lab-status-dot');
  if (d) d.className = 'lab-dot err';
};
const stillStarting = () => { const s = document.getElementById('lab-status-text'); return s && /^(starting|loading|building)/.test(s.textContent); };
window.addEventListener('error', (ev) => { if (stillStarting()) fatal(ev.error || ev.message); });
window.addEventListener('unhandledrejection', (ev) => { if (stillStarting()) fatal(ev.reason); });

(async () => {
  try {
    const status = document.getElementById('lab-status-text');
    if (status) status.textContent = 'loading modules …';
    const [{ loadCore }, { Lab }] = await Promise.all([import('./hcc.js'), import('./lab/app.js')]);
    const corePromise = loadCore();
    corePromise.catch(() => {});          // reported through lab.start()
    const lab = new Lab();
    await lab.start(corePromise);
  } catch (e) { fatal(e); }
})();
