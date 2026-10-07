// Stub entry point; the real application will replace this file.
const el = document.getElementById("status");
const msg = "Interactive Lab stub: main.js loaded. WebGPU " +
  ("gpu" in navigator ? "available." : "not available (Canvas2D fallback).");
console.log(msg);
if (el) el.textContent = msg;
