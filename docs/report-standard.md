# Interactive scientific report: the standard

Version 0.1, 2026-10-09. Maintained as the README of https://github.com/yipihey/quadrant; this copy is the
snapshot the book was built from. This document specifies the format that the
Hopf–Cole Cosmology book follows and that new reports should follow. The
companion skill (`.claude/skills/interactive-report/SKILL.md`) tells an agent
how to work inside it; `docs/format-audit.md` records why these choices were
made.

## 1. Shape of a report

A report is a repository with four parts and one pipeline.

```
_quarto.yml  _brand.yml  CITATION.cff  Makefile  README.md
chapters/    narrative (Quarto book chapters), references.bib, notation.qmd, status.qmd
core/        the numerical core: a tested library (Rust by default) compiled to WebAssembly
app/         the standalone laboratory and the chapter widgets (plain ES modules, no bundler)
docs/        decisions, audits, this standard
_extensions/hcc/   the Quarto extension: provenance and lab shortcodes, their CSS
scripts/     build_info.py, brand_to_css.py, check_links.py, serve.py
.github/workflows/publish.yml   build core → docs → check → render → deploy
```

The pipeline is one workflow: build the WebAssembly, build the API
documentation into `api/`, check the provenance links, render the book,
deploy. Locally `make all` does the same; `make check` runs the link check.

## 2. The text

- A Quarto **book** (not a website): chapters cross-reference each other's
  equations and sections, so every equation that is referred to has a
  label (`{#eq-…}`) and every section that is linked has an explicit id
  (`{#sec-…}`).
- Numbered sections to depth 2, numbered equations, KaTeX math, full-text
  search, light and dark themes, `code-fold` and `code-tools` (readers can
  see the chapter source), `repo-actions: [edit, source, issue]`,
  Hypothesis annotations, `date-modified: last-modified`.
- Two appendices are mandatory: **Notation and conventions** (symbols, units,
  signs, clocks, estimator windows) and **Status** (build revision, what is
  verified and where, lineage, how to cite, how to share and annotate).
- Every chapter that introduces an experiment has a **"Try it" callout**
  ending in a `{{< lab >}}` button whose hash reproduces the experiment.
- Every citation is verified against ADS (or the field's equivalent) before
  it enters `references.bib`; keep the bibcode or DOI in the entry.
- Prose names a file or function only when the reader has to go there;
  provenance links carry the rest (section 4).

## 3. The laboratory and the widgets

- The app is standalone: served from `app/` with its own `index.html`,
  plain ES modules, the WebAssembly package in `app/pkg/`, no build step
  beyond `wasm-pack`. The book copies it verbatim (`resources: app/**`).
- **All state lives in the URL hash**, encoded by one schema
  (`app/js/lab/state.js`: `[stateKey, hashKey, type, range]`). Only
  non-default values are written. A hash is therefore a permalink, a
  preset, a regression fixture and a deep-link target at once.
- A **presets** list (`app/js/lab/presets.js`) names the experiments of the
  lab guide by their hashes.
- Every panel has a caption, a "§ book" back-link to the section that
  explains it, an `aria-label`, and an **Explain** note in the app.
- **Plots** are theme-aware SVG from one class (`app/js/viz/plot.js`):
  interactive (wheel zoom, drag pan, double-click reset, lin/log toggles,
  typed ranges) and exportable (SVG, CSV). Field panels export PNG. A
  copy-link button copies the permalink.
- **Capability tiers**: WebGPU compute and rendering when available,
  Canvas2D and WebAssembly fallbacks, a lite mode for weak GPUs, a
  self-test page that validates the GPU path against the WebAssembly
  reference. A head include marks the document with the detected tier.
- Chapter widgets use one loader contract: `<div class="hcc-widget"
  data-widget="name"></div>` plus one module script per page.

## 4. Provenance: text ↔ code ↔ tests

Three shortcodes, provided by the extension, make the connection explicit
and checkable:

| Shortcode | Renders | Resolves against |
|---|---|---|
| `{{< src module::item >}}` | subtle superscript ⟨item⟩ linking to the rustdoc page | `api/` (built by `make api`) |
| `{{< test test_name "label" >}}` | badge "✓ label" linking to the test source | `core/tests/*.rs` |
| `{{< lab "hash" "label" >}}` | button opening the lab with that state | `app/js/lab/state.js` keys |

Rules:

1. Every equation the code evaluates carries a `src` link in the sentence
   that introduces it. Every quantitative claim ("agrees to 10⁻³", "mass is
   conserved") carries a `test` badge naming the test that pins it.
2. The rustdoc of the core is published with the report (`api/`); doc
   comments are therefore part of the publication and are written for
   readers, with the equation numbers of the text.
3. `scripts/check_links.py` fails the build when a `src` target has no
   page, a `test` name has no function, or a `lab` hash uses an unknown
   key.

## 5. Appearance

- `_brand.yml` holds the palette (and, optionally, fonts and logo). Quarto
  applies it to the book; `scripts/brand_to_css.py` turns it into
  `app/css/brand.css` (CSS custom properties, `--brand-*` and the app's
  `--hcc-accent`) so the laboratory uses the same colours. Change colours in
  one file only.
- Theme-dependent styling uses `currentColor` and CSS variables; nothing
  hard-codes a background. Plots inherit the page colour.

## 6. Sharing

- Footer: build revision and date (`scripts/build_info.py` →
  `_variables.yml` → `{{< var build.sha >}}`).
- `CITATION.cff` in the repository; a DOI (Zenodo) once released; the
  status appendix shows "cite as".
- Hypothesis for annotations; GitHub issues and "Edit this page" for
  corrections. Giscus (Discussions) is optional and needs the repository's
  Discussions enabled.
- Lab permalinks, SVG/CSV/PNG exports, named presets.
- The original notebooks or data that the report grew from are included and
  linked from the status appendix.

## 7. Verification before publication

The report is publishable when all of the following hold:

1. `cargo test --release` passes and every `test` badge names a passing
   test.
2. `make check` passes (provenance links resolve).
3. `quarto render` produces no unresolved cross-reference or citation
   warnings.
4. The rendered pages have been looked at in a browser: equation widths,
   callouts, provenance marks, lab deep links, console errors in the app.
5. CI is green and the deployed site loads the laboratory.

## 8. Reusing the standard

To start a new report, copy `_extensions/hcc/`, `scripts/`, `includes/`,
`Makefile`, `.github/workflows/publish.yml`, `_brand.yml`, the two
appendices and the `_quarto.yml` skeleton; replace the core and the app's
physics; keep the conventions of sections 2 to 7. The skill automates the
checklist.
