// Entry point of the Interactive Laboratory.
import { loadCore } from './hcc.js';
import { Lab } from './lab/app.js';

const fatal = (e) => {
  console.error(e);
  const s = document.getElementById('lab-status-text');
  if (s) { s.textContent = 'Failed to start: ' + (e && e.message ? e.message : e); s.className = 'err'; }
};

(async () => {
  try {
    const status = document.getElementById('lab-status-text');
    if (status) status.textContent = 'loading WebAssembly core …';
    const core = await loadCore();
    const lab = new Lab(core);
    await lab.start();
  } catch (e) { fatal(e); }
})();
