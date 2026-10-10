---
name: interactive-report
description: Build or extend an interactive scientific report in the Hopf–Cole Cosmology format (Quarto book + tested Rust/WASM core + standalone browser lab) with provenance links, deep links, verified citations and a release checklist. Use when adding a chapter, an equation, a lab panel or a new report in this format.
---

# Interactive scientific report

The format is specified in `docs/report-standard.md`; this skill is the working
procedure. Read the standard once, then follow the steps below. The audit that
motivated the format is `docs/format-audit.md`.

## Layout you can rely on

- `chapters/*.qmd` narrative; `chapters/references.bib`; appendices
  `notation.qmd`, `status.qmd`.
- `core/` tested numerical core (Rust → WASM via `wasm-pack build core
  --target web --release --out-dir ../app/pkg`); tests in `core/tests/`.
- `app/` standalone lab and widgets; all lab state in the URL hash, schema in
  `app/js/lab/state.js`; presets in `app/js/lab/presets.js`; explain notes in
  `app/js/lab/explain.js`; panel catalog with captions and `doc` back-links in
  `app/js/lab/catalog.js`; shared interactive plot in `app/js/viz/plot.js`.
- `_extensions/yipihey/quadrant/` shortcodes `{{< src >}}`, `{{< test >}}`, `{{< lab >}}`
  (the quadrant extension, https://github.com/yipihey/quadrant; configured under `quadrant:` in `_quarto.yml`).
- `make wasm api check site test`; `python3 scripts/serve.py 8790 _site` for a
  no-cache dev server; `rsync -a app/ _site/app/` to refresh the app copy
  without re-rendering.

## Adding physics: derive, test, write, link

1. **Derive** the result and check it numerically in the core before any text
   exists (a Rust test, or an example under `core/examples/`). The test must
   pin the claim you intend to make ("matches to 10⁻³", "mass conserved").
2. **Expose** what the lab needs as a WASM export in `core/src/wasm.rs`;
   rebuild with `wasm-pack`.
3. **Write** the section: pedagogy first, equations labelled `{#eq-…}`,
   sections that will be linked get `{#sec-…}`. Keep code names out of prose;
   attach `{{< src module::item >}}` to the sentence that introduces an
   equation the code evaluates and `{{< test name "label" >}}` to every
   quantitative claim. Sentence-end punctuation goes after the badge; never
   glue a shortcode to a `@ref` (leave a word between them).
4. **Cite** only after verifying the reference in ADS (or the field's index);
   keep volume, pages, DOI or bibcode in the `.bib` entry.
5. **Link the lab**: a "Try it" callout ending in `{{< lab "hash" "label" >}}`
   whose hash reproduces the experiment (use the schema keys; lists and vectors are joined with `_`, e.g.
   `se=lin_sheet_hc`, `k2=3_0_0`; slots are `kind.sub.cmap.flag` joined by `_`). Add the experiment to `presets.js` and the lab guide.
6. **Explain in the app**: a note in `explain.js`, a caption with a `doc`
   back-link in `catalog.js`, a row in the lab-guide table.
7. **Update the notation appendix** if a symbol or convention is new.

## Adding a lab panel or plot

- Fields go through the Engine cache keyed by the applied parameters; never
  read the WASM sim's "last result" lazily. Mark panels stale while
  recomputing; run heavy work as tasks in the section's task list.
- Plots: `new LinePlot(host, {width, height})`, `setAxes`, `setSeries`,
  `setMarkers`, `draw()`. The toolbar, export and interactivity come for free.
- Every new control gets one hash key in `state.js` (short, documented in the
  SCHEMA comment); defaults are not written to the hash.
- GPU code needs a WASM or Canvas2D fallback and a self-test entry.

## Delegation

Delegate code with a brief that states the physics, the verified numbers, the
files to read and the exact browser checks to perform; keep derivations and
chapter text with the stronger model. Agents must not edit `.qmd` files,
`_quarto.yml` or `core/` unless the brief says so, and must report what they
could not verify.

## Release checklist (all must hold before "done")

1. `cargo test --release` green; the badges name passing tests.
2. `make api && python3 scripts/check_links.py` → "provenance links: OK".
3. `quarto render` with no unresolved cross-reference or citation warning.
4. Looked at the rendered pages in a browser: equation widths, callouts,
   provenance marks, lab buttons; the lab console has no new errors; the
   presets load.
5. Commit with a message that names the physics, the lab change and the text
   change; push; CI green; the deployed site opens the lab.
6. Record non-derivable conventions (signs, clocks, normalisations, decisions)
   in `docs/` and in memory.
