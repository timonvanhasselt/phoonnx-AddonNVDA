#!/usr/bin/env python3
"""Build the .nvda-addon package.

An NVDA add-on is a zip file containing manifest.ini and the add-on
directories. The phoonnx runtime and its dependencies must be vendored into
synthDrivers/phoonnx/phoonnx_libs/ and the .onnx voice weights placed next to
the driver before building a runnable package; this script warns (but does
not fail) when they are absent so CI can still produce a skeleton artifact.

Usage: python build_addon.py [output_dir]
"""
import configparser
import os
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {".git", ".github", "test", "__pycache__"}
EXCLUDE_FILES = {"build_addon.py", "TODO.md", "AGENTS.md", "CLAUDE.md"}


def read_manifest():
    parser = configparser.ConfigParser()
    with open(os.path.join(REPO_ROOT, "manifest.ini"), encoding="utf-8") as f:
        parser.read_string("[__root__]\n" + f.read())
    root = parser["__root__"]
    return root.get("name", "phoonnx").strip('"'), root.get("version", "0.0").strip('"')


def iter_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn in EXCLUDE_FILES or fn.startswith(".") or fn.endswith((".pyc", ".nvda-addon")):
                continue
            path = os.path.join(dirpath, fn)
            yield path, os.path.relpath(path, REPO_ROOT)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT
    _, version = read_manifest()
    out_path = os.path.join(out_dir, f"phoonnx-{version}.nvda-addon")

    driver_dir = os.path.join(REPO_ROOT, "synthDrivers", "phoonnx")
    if not os.path.isdir(os.path.join(driver_dir, "phoonnx_libs")):
        print("WARNING: synthDrivers/phoonnx/phoonnx_libs/ missing — "
              "package will not run until phoonnx deps are vendored in.")
    if not any(f.endswith(".onnx") for f in os.listdir(driver_dir)):
        print("WARNING: no .onnx voice model found — "
              "package will not pass the driver check() until one is added.")

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in iter_files():
            zf.write(path, arcname)
    print(f"Built {out_path}")


if __name__ == "__main__":
    main()
