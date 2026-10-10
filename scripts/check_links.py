#!/usr/bin/env python3
"""Check the provenance shortcodes of the text: every {{< src >}} resolves to a rustdoc page under api/, every
{{< test >}} names a test function under core/tests, and every {{< lab >}} hash uses known state keys and the '_'
list separator. Exit 1 on failure. Configuration is read from _quarto.yml (quadrant: crate/api/tests)."""
import re, sys, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
KINDS = ["struct", "enum", "trait", "type", "union"]

cfg = {"api": "api", "tests": "core/tests", "crate": None}
qy = (ROOT / "_quarto.yml").read_text() if (ROOT / "_quarto.yml").exists() else ""
m = re.search(r"^quadrant:\s*\n((?:[ \t]+.*\n?)+)", qy, re.M)
if m:
    for line in m.group(1).splitlines():
        k, _, v = line.strip().partition(":")
        if k in cfg and v.strip():
            cfg[k] = v.strip().strip('"').strip("'")
API = ROOT / cfg["api"]
if not cfg["crate"]:
    cands = [d.name for d in API.iterdir() if d.is_dir() and (d / "index.html").exists() and d.name not in ("src", "static.files")] if API.exists() else []
    cfg["crate"] = cands[0] if cands else "core"
CRATE_DIR = API / cfg["crate"]

def resolve_src(path):
    parts = path.split("::")
    item = parts[-1]
    if item[:1].isupper():
        d = CRATE_DIR.joinpath(*parts[:-1])
        return any((d / f"{k}.{item}.html").exists() for k in KINDS)
    if len(parts) >= 2 and parts[-2][:1].isupper():
        d = CRATE_DIR.joinpath(*parts[:-2])
        for k in KINDS:
            f = d / f"{k}.{parts[-2]}.html"
            if f.exists():
                return f'id="method.{item}"' in f.read_text(errors="ignore")
        return False
    d = CRATE_DIR.joinpath(*parts[:-1])
    return (d / f"fn.{item}.html").exists() or (d / item / "index.html").exists()

tests_src = "\n".join(p.read_text() for p in (ROOT / cfg["tests"]).glob("*.rs")) if (ROOT / cfg["tests"]).exists() else ""
keys = set()
for sj in glob.glob(str(ROOT / "app" / "js" / "**" / "state.js"), recursive=True):
    keys |= set(re.findall(r"\['[A-Za-z0-9]+', '([A-Za-z0-9]+)', '", pathlib.Path(sj).read_text()))

bad = 0
files = sorted(glob.glob(str(ROOT / "chapters" / "*.qmd"))) + sorted(glob.glob(str(ROOT / "*.qmd")))
for qmd in files:
    text = pathlib.Path(qmd).read_text()
    rel = pathlib.Path(qmd).relative_to(ROOT)
    for mm in re.finditer(r"\{\{<\s*src\s+([A-Za-z0-9_:]+)", text):
        if not resolve_src(mm.group(1)):
            print(f"{rel}: src {mm.group(1)} has no rustdoc page under {CRATE_DIR.relative_to(ROOT)}"); bad += 1
    for mm in re.finditer(r"\{\{<\s*test\s+([A-Za-z0-9_]+)", text):
        if not re.search(rf"fn\s+{mm.group(1)}\s*\(", tests_src):
            print(f"{rel}: test {mm.group(1)} not found under {cfg['tests']}"); bad += 1
    for mm in re.finditer(r"\{\{<\s*lab\s+\"([^\"]*)\"", text):
        if "%2C" in mm.group(1) or "," in mm.group(1):
            print(f"{rel}: lab hash {mm.group(1)!r} uses commas; lists are joined with '_'"); bad += 1
        if keys:
            for kv in mm.group(1).split("&"):
                if kv and kv.split("=")[0] not in keys:
                    print(f"{rel}: lab hash key {kv.split('=')[0]!r} is not in the state schema"); bad += 1
print("provenance links:", "OK" if bad == 0 else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
