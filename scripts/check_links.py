#!/usr/bin/env python3
"""Check the provenance shortcodes of the chapters: every {{< src >}} resolves to a rustdoc page in api/, every
{{< test >}} names a test function in core/tests, and every {{< lab >}} hash uses known state keys. Exit 1 on failure."""
import re, sys, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CRATE = "hcc_core"
KINDS = ["struct", "enum", "trait", "type", "union"]

def resolve_src(path):
    parts = path.split("::")
    root = ROOT / "api" / CRATE
    item = parts[-1]
    if item[:1].isupper():
        d = root.joinpath(*parts[:-1])
        return any((d / f"{k}.{item}.html").exists() for k in KINDS)
    if len(parts) >= 2 and parts[-2][:1].isupper():
        d = root.joinpath(*parts[:-2])
        for k in KINDS:
            f = d / f"{k}.{parts[-2]}.html"
            if f.exists():
                return f'id="method.{item}"' in f.read_text(errors="ignore")
        return False
    d = root.joinpath(*parts[:-1])
    return (d / f"fn.{item}.html").exists() or (d / item / "index.html").exists()

tests_src = "\n".join(p.read_text() for p in (ROOT / "core" / "tests").glob("*.rs"))
state_js = (ROOT / "app" / "js" / "lab" / "state.js").read_text()
keys = set(re.findall(r"\['[A-Za-z0-9]+', '([A-Za-z0-9]+)', '", state_js))

bad = 0
for qmd in sorted(glob.glob(str(ROOT / "chapters" / "*.qmd")) + [str(ROOT / "index.qmd")]):
    text = pathlib.Path(qmd).read_text()
    for m in re.finditer(r"\{\{<\s*src\s+([A-Za-z0-9_:]+)", text):
        if not resolve_src(m.group(1)):
            print(f"{qmd}: src {m.group(1)} has no rustdoc page"); bad += 1
    for m in re.finditer(r"\{\{<\s*test\s+([A-Za-z0-9_]+)", text):
        if not re.search(rf"fn\s+{m.group(1)}\s*\(", tests_src):
            print(f"{qmd}: test {m.group(1)} not found in core/tests"); bad += 1
    for m in re.finditer(r"\{\{<\s*lab\s+\"([^\"]*)\"", text):
        if "%2C" in m.group(1) or "," in m.group(1):
            print(f"{qmd}: lab hash {m.group(1)!r} uses commas; lists are joined with '_'"); bad += 1
        for kv in m.group(1).split("&"):
            if kv and kv.split("=")[0] not in keys:
                print(f"{qmd}: lab hash key {kv.split('=')[0]!r} is not in the state schema"); bad += 1
print("provenance links:", "OK" if bad == 0 else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
