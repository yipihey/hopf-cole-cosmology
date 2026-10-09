#!/usr/bin/env python3
"""Write _variables.yml with the git revision and build date, used in the page footer and on the status page
({{< var build.sha >}} etc.). Runs as a Quarto pre-render step."""
import pathlib, subprocess, datetime
ROOT = pathlib.Path(__file__).resolve().parent.parent
def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""
sha = git("rev-parse", "--short=9", "HEAD") or "unknown"
full = git("rev-parse", "HEAD") or ""
date = git("log", "-1", "--format=%cs") or datetime.date.today().isoformat()
dirty = " (uncommitted changes)" if git("status", "--porcelain") else ""
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
(ROOT / "_variables.yml").write_text(f"""build:
  sha: "{sha}{dirty}"
  commit: "{full}"
  date: "{date}"
  rendered: "{now}"
""")
print("build:", sha + dirty, date)
