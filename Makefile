.PHONY: wasm site preview test all

wasm:
	wasm-pack build core --target web --release --out-dir ../app/pkg

site:
	quarto render

preview:
	quarto preview

test:
	cargo test --release --manifest-path core/Cargo.toml

all: wasm site

# Local Python environment for executing the chapter code cells (figures are
# frozen in _freeze/, so this is only needed when editing those cells).
kernel:
	uv venv .venv && uv pip install -p .venv/bin/python numpy scipy matplotlib sympy jupyter ipykernel
	.venv/bin/python -m ipykernel install --user --name hcc --display-name "hcc (venv)"
	echo "QUARTO_PYTHON=$(PWD)/.venv/bin/python" > _environment.local
