# Interactive scientific reports: audit of this project and the case for a standard

Status: draft, 2026-10-09. Written for the Hopf–Cole Cosmology book, as the
first step towards a reusable format ("skill" plus template) for agent-built
scientific reports.

## 1. What we have

**Quarto book** (1.10), HTML only. Nine chapters plus references, numbered
sections and equations with about 140 cross-references, KaTeX math, a
verified bibliography (every entry checked against ADS), full-text search,
docked sidebar, light and dark Bootstrap themes (cosmo/darkly), `code-fold`,
`freeze: auto` for the four Python figure cells, one custom stylesheet
(`app/css/hcc.css`), a head include that detects WebGPU, and a GitHub issue
link on every page.

**Standalone app** (`app/`) copied verbatim into the site: the 2D/3D
laboratory and the chapter widgets, driven by a Rust core compiled to
WebAssembly, with a WebGPU compute path, a Canvas2D fallback and a lite
mode. The whole lab state is encoded in the URL hash by one schema
(`app/js/lab/state.js`), so every view is already a permalink. Plots are
SVG, theme-aware, and now interactive (zoom, pan, lin/log, ranges).

**Rust core** (`core/`) with a validation suite (27 tests) that pins the
physics the text claims: nLPT recursion by equation-of-motion residuals,
growth functions, exact deposits against fixtures from the Julia port, the
viscous kernel, phase statistics. A GPU self-test page checks the WebGPU
path against the WASM reference.

**Pipeline**: one GitHub Actions workflow builds the WASM, renders the book
and deploys to Pages. Local: `make wasm site test`, a no-cache dev server.

Strengths worth keeping in any standard: the text/lab/core triangle with
tests as ground truth; hash-encoded state; verified citations; theme-aware
SVG plots; explain notes and a lab guide written alongside the code.

## 2. Gaps and recommendations

### 2.1 Provenance: connecting the text to the code that computes it

This is the largest gap. The chapters mention source files in prose but
nothing is clickable, and line-number links rot.

1. **Publish the Rust API documentation inside the site.** `cargo doc
   --no-deps` in CI, copied to `_site/api/`. Rustdoc anchors are stable
   (`api/hcc_core/spectra/fn.f2_viscous.html`), so links never rot, and the
   doc comments become part of the publication.
2. **Two shortcodes, implemented as a small Quarto extension**
   (`_extensions/hcc/`):
   - `{{< src spectra::f2_viscous >}}` renders a subtle superscript glyph
     after an equation that links to the rustdoc item (and, on hover, shows
     the signature).
   - `{{< test viscous_kernel_limits_and_two_plane_waves >}}` renders a
     "verified by" badge linking to the test source and, if we publish the
     `cargo test` log as a page, to the actual numbers.
   The convention is then: every equation the code evaluates carries a
   `src` link, every quantitative claim ("agrees to 10⁻³") carries a `test`
   link. Subtle in the text, mechanical to check.
3. **Deep links both ways.** Every "Try it" callout gets a button whose
   href is the exact lab hash for that experiment; every lab panel caption
   links to the chapter section that explains it. The hash schema makes the
   first trivial; the second needs one `href` per panel kind in the catalog.
4. **"View source" for the chapter itself** is one line (`code-tools:
   true`), which gives readers the `.qmd`.

### 2.2 Appearance

The light/dark toggle exists. What is missing is a single place to set
typography, widths and colours for the book *and* the app:

- Use Quarto's `brand.yml` (supported since 1.6) for colours, fonts and
  logo, and read the same values into the app's CSS variables
  (`--hcc-accent`, `--hcc-border`, …) so the lab matches the book.
- Expose a small `styles/theme.scss` with SCSS variables for the Bootstrap
  theme (font family, base size, content width) rather than overriding in
  `hcc.css`.
- A reader-side font-size control is unnecessary; browsers do this.

### 2.3 Sharing and collaboration

Cheap and valuable, roughly in order:

1. `repo-actions: [edit, source, issue]` — "Edit this page" links.
2. Comments: `comments: giscus` (GitHub Discussions) or `hypothesis: true`
   (anchored annotations, good for referee-style remarks). One line each.
3. Citation: `CITATION.cff` plus a Zenodo DOI on release, and Quarto's
   `citation:` metadata so the page carries "cite as".
4. Versioning in the footer: `date: last-modified` and the git SHA (a
   pre-render script writes it to a variable), so a screenshot or quote can
   be tied to a build.
5. **Export from the lab**: copy-permalink button (the hash), SVG download
   of any plot (`toSVGString` exists), CSV/JSON of spectra, PDFs and kernel
   tables, PNG of any field panel. Scientists share numbers, not only
   pictures.
6. A **named-experiment gallery**: the lab guide's experiments as links with
   their hashes, which doubles as regression fixtures for the agents.
7. Notebook links: the original lecture notebook is in `notebooks/`; embed
   its figures with `{{< embed >}}` and offer the download so the lineage
   from lecture to book is visible.
8. A notation appendix (symbols, conventions such as D as time, units) and
   a short "status" page: test count, GPU self-test link, toolchain
   versions, last build.

### 2.4 Housekeeping

- No PDF format. Interactive content does not survive print; if a PDF is
  wanted, give widgets a static fallback image and add `format: pdf` later.
- `app/pkg` is a build product; fine, but pin the `wasm-pack` version in CI.
- Accessibility: field canvases need `aria-label`s; the plots already have
  `role="img"`.

## 3. What to turn into a standard

The reusable part is not this physics; it is the shape of the project. The
standard should be three things:

**A. A Quarto extension** (`hcc-report`, or a neutral name) providing:
- the `src`, `test` and `lab` shortcodes (provenance links and deep links),
- the theme (`brand.yml`, SCSS variables, `hcc.css` with CSS variables),
- the widget loader contract: `<div class="xxx-widget" data-widget="name">`
  plus one module script per page,
- the interactive SVG plot class, colormaps and colorbar components,
- the head include (capability detection) and lite-mode conventions.

**B. A project template** (`quarto use template …`) with the skeleton:
`_quarto.yml` book, `chapters/`, `app/` (standalone, hash-encoded state,
GPU fallback, self-test page), `core/` (a tested numerical core compiled
to WASM, Rust by default), `docs/`, the CI workflow, the Makefile, the
no-cache dev server.

**C. A skill** that tells an agent how to work in that template. The rules
that made this project work, which belong in the skill rather than in the
tooling:
1. Derive first, test second, write third: no equation goes into the text
   until a core test pins it; the test name goes into the text.
2. Every citation verified against ADS before it enters the `.bib`; keep the
   bibcode in the entry.
3. Every "Try it" has a deep link; every lab panel has an explain note and a
   lab-guide row; every control has a hash key documented in one schema.
4. Render and look at the page in a browser before committing (equation
   width, cross-references, console errors); CI must be green before
   reporting.
5. Keep a decisions log (`docs/`) for conventions that are not derivable from
   the code (sign conventions, clocks, normalisations).
6. Delegate code to cheaper models with a brief that states the physics, the
   verified numbers and the files; keep derivations and text with the
   stronger model.

## 4. Suggested order

1. rustdoc in the site + `src`/`test` shortcodes + deep-linked "Try it"
   buttons (provenance; one day).
2. `brand.yml` + SCSS variables + `repo-actions` + giscus/hypothesis +
   `CITATION.cff` + footer SHA (sharing; half a day).
3. Lab export buttons (permalink, SVG, CSV, PNG) and the experiment gallery
   (half a day).
4. Factor the extension and template out of this repo, write the skill
   (one to two days), then re-apply to this book as its first user.
