#!/usr/bin/env python3
"""Generate app/css/brand.css (CSS custom properties) from _brand.yml so the standalone lab shares the book's palette.
Runs as a Quarto pre-render step and from `make brand`; a minimal YAML reader is used so no dependency is needed."""
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
src = (ROOT / "_brand.yml").read_text().splitlines()
palette, roles, section = {}, {}, None
for line in src:
    s = line.rstrip()
    if not s.strip() or s.lstrip().startswith("#"):
        continue
    indent = len(s) - len(s.lstrip())
    key, _, val = s.strip().partition(":")
    val = val.strip().strip('"').strip("'")
    if indent == 0:
        section = key
    elif section == "color" and indent == 2 and key == "palette":
        pass
    elif section == "color" and indent == 4:
        palette[key] = val
    elif section == "color" and indent == 2 and val:
        roles[key] = palette.get(val, val)
out = ["/* generated from _brand.yml by scripts/brand_to_css.py; do not edit */", ":root {"]
for k, v in palette.items():
    out.append(f"  --brand-{k}: {v};")
for k, v in roles.items():
    out.append(f"  --brand-{k}: {v};")
if "primary" in roles:
    out.append(f"  --hcc-accent: {roles['primary']};")
out.append("}")
if "secondary" in roles:
    out += ['@media (prefers-color-scheme: dark) { :root:not([data-bs-theme="light"]):not(.quarto-light) { --hcc-accent: ' + roles["secondary"] + "; } }",
            ':root[data-bs-theme="dark"], :root.quarto-dark { --hcc-accent: ' + roles["secondary"] + "; }"]
(ROOT / "app" / "css" / "brand.css").write_text("\n".join(out) + "\n")
print("brand.css:", ", ".join(f"{k}={v}" for k, v in roles.items()))
