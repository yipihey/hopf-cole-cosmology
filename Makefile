.PHONY: wasm api check brand site preview test all

wasm:
	wasm-pack build core --target web --release --out-dir ../app/pkg

# Rust API documentation, copied into the site as api/ (the {{< src >}} shortcodes link into it)
api:
	cargo doc --no-deps --release --manifest-path core/Cargo.toml
	rm -rf api && cp -r core/target/doc api

# Provenance links of the chapters resolve (needs api/)
check: api
	python3 scripts/check_links.py

# app/css/brand.css from _brand.yml (also run by quarto as a pre-render step)
brand:
	python3 scripts/brand_to_css.py

site:
	quarto render

preview:
	quarto preview

test:
	cargo test --release --manifest-path core/Cargo.toml

all: wasm api site

# Local Python environment for executing the chapter code cells (figures are
# frozen in _freeze/, so this is only needed when editing those cells).
kernel:
	uv venv .venv && uv pip install -p .venv/bin/python numpy scipy matplotlib sympy jupyter ipykernel
	.venv/bin/python -m ipykernel install --user --name hcc --display-name "hcc (venv)"
	echo "QUARTO_PYTHON=$(PWD)/.venv/bin/python" > _environment.local
