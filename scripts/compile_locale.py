#!/usr/bin/env python3
"""Compile all .po files to .mo files.

Requires msgfmt on PATH.
  Windows: choco install gettext  OR  pip install Babel (provides pybabel)
  Linux/macOS: apt/brew install gettext
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOCALE_DIR = ROOT / "locale"

for po_file in LOCALE_DIR.rglob("*.po"):
    mo_file = po_file.with_suffix(".mo")
    print(f"Compiling {po_file.relative_to(ROOT)} -> {mo_file.relative_to(ROOT)}")
    try:
        subprocess.run(["msgfmt", str(po_file), "-o", str(mo_file)], check=True)
    except FileNotFoundError:
        print("ERROR: msgfmt not found. Install gettext (choco install gettext) "
              "or Babel (pip install Babel and use pybabel compile).", file=sys.stderr)
        sys.exit(1)

print("Done.")
