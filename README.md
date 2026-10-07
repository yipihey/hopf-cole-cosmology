# Hopf–Cole Cosmology

The Hopf–Cole transform maps Burgers' equation, a nonlinear advection–diffusion equation that forms shocks, onto the linear heat equation. This project is an online text that develops that correspondence from first principles and shows how it underlies the Zel'dovich approximation, the adhesion model, and Lagrangian perturbation theory (LPT) for cosmological structure formation.

The numerical core is a Rust crate (`hcc-core`) compiled to WebAssembly, driving interactive 2D/3D experiments in the browser with WebGPU visualization (falling back to Canvas2D). The text is a Quarto website deployed to GitHub Pages.

**Live site:** <https://yipihey.github.io/hopf-cole-cosmology/>

## Building locally

Requirements: [Rust](https://rustup.rs) with the `wasm32-unknown-unknown` target, [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), and [Quarto](https://quarto.org) (1.10 or later).

```sh
rustup target add wasm32-unknown-unknown
make wasm      # build the WASM module into app/pkg/
make site      # render the website into _site/
make preview   # live-preview the website
make test      # run the Rust tests
make all       # wasm + site
```

The WASM module must be served over HTTP (not `file://`), e.g. via `make preview` or `python3 -m http.server -d _site`.

## Repository layout

- `core/` — Rust crate `hcc-core` (solvers, kernels, LPT), compiled to WASM
- `app/` — standalone interactive lab (`index.html`, `js/`, `css/`, `shaders/`; `pkg/` is generated); copied verbatim into the site
- `chapters/` — Quarto chapter sources and `references.bib`
- `index.qmd`, `_quarto.yml` — site landing page and configuration
- `includes/` — HTML snippets injected into every page
- `notebooks/` — exploratory Jupyter notebooks
- `.github/workflows/publish.yml` — build and deploy to GitHub Pages

## License

MIT; see [LICENSE](LICENSE).
