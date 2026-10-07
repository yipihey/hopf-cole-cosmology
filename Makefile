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
