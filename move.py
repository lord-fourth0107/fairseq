#!/usr/bin/env python3
"""
move.py — Copy a directory tree on HPC with visible progress.

Usage:
  python move.py SRC DST [--mirror] [--dry-run] [--checksum] [--keep-owners]

Notes:
- Trailing slash on SRC (…/src/): copies the *contents* into DST (rsync semantics).
- No trailing slash (…/src): nests 'src' inside DST.
- By default we pass --no-owner --no-group to avoid chown/chgrp errors on shared HPC filesystems.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

def rsync_available() -> bool:
    try:
        subprocess.run(["rsync", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def run_rsync(src: str, dst: str, mirror=False, dry_run=False, checksum=False, keep_owners=False):
    """
    Run rsync with **live progress** to TTY (no stdout capture).
    """
    # Base options
    cmd = ["rsync", "-aH", "-h", "--partial"]
    # Ownership/group handling (default: don't try to chown/chgrp on HPC)
    if not keep_owners:
        cmd += ["--no-owner", "--no-group"]

    if mirror:
        cmd.append("--delete")
    if checksum:
        cmd.append("--checksum")

    # Progress / verbosity
    if dry_run:
        # Show what WOULD copy, with filenames + stats
        cmd += ["--dry-run", "-v", "--info=NAME,stats1"]
    else:
        # -P gives per-file progress; --info=progress2 adds total transfer progress
        cmd += ["-P", "--info=progress2"]

    # Ensure destination exists
    Path(dst).mkdir(parents=True, exist_ok=True)

    # Run without capturing output so progress renders live
    full_cmd = cmd + [src, dst]
    # Helpful for debugging: print the exact rsync command (comment out if noisy)
    print(">>", " ".join(shlex_quote(a) for a in full_cmd))
    try:
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rsync failed with exit code {e.returncode}") from e

def shlex_quote(s: str) -> str:
    # Minimal portable shell quoting for the printed command line
    if not s or any(c.isspace() or c in "\"'\\$`" for c in s):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s

def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def copy_with_progress_python(src: str, dst: str):
    """
    Pure-Python fallback with simple progress.
    Not resumable and slower than rsync; use only if rsync is unavailable.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    # rsync-like trailing slash semantics
    copy_contents = src.endswith(os.sep) and src_path.is_dir()

    files = list(iter_files(src_path)) if src_path.is_dir() else [src_path]
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    copied_bytes = 0

    for f in files:
        rel = f.relative_to(src_path) if src_path.is_dir() else Path(f.name)
        out = (
            (dst_path / rel) if copy_contents
            else ((dst_path / src_path.name / rel) if src_path.is_dir() else (dst_path / rel))
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        if f.is_symlink():
            target = os.readlink(f)
            if out.exists():
                out.unlink()
            os.symlink(target, out)
            continue

        # Copy in chunks and print total progress
        with open(f, "rb") as rf, open(out, "wb") as wf:
            while True:
                chunk = rf.read(1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
                copied_bytes += len(chunk)
                pct = (copied_bytes / total_bytes * 100.0) if total_bytes else 100.0
                print(f"\rCopying... {copied_bytes:,}/{total_bytes:,} bytes ({pct:5.1f}%)", end="", flush=True)
        shutil.copystat(f, out, follow_symlinks=False)
    print()  # newline after progress

def main():
    ap = argparse.ArgumentParser(description="Copy a folder to another location with visible progress (HPC-friendly).")
    ap.add_argument("source", help="Source path (dir or file). Use trailing slash to copy contents.")
    ap.add_argument("destination", help="Destination directory.")
    ap.add_argument("--mirror", action="store_true", help="Mirror mode: delete extras in destination.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without copying.")
    ap.add_argument("--checksum", action="store_true", help="Use checksums to detect changes (slower, safer).")
    ap.add_argument("--keep-owners", action="store_true",
                    help="Keep owner/group (omit --no-owner/--no-group). Use only if you control dest ownership.")
    ap.add_argument("--force-python", action="store_true", help="Force pure-Python fallback (no rsync).")
    args = ap.parse_args()

    src = args.source  # keep user-provided trailing slash semantics
    dst = os.path.abspath(args.destination)

    try:
        if not args.force_python and rsync_available():
            run_rsync(src, dst, mirror=args.mirror, dry_run=args.dry_run,
                      checksum=args.checksum, keep_owners=args.keep_owners)
        else:
            if args.mirror or args.dry_run or args.checksum:
                print("[!] --mirror/--dry-run/--checksum aren't supported in pure-Python mode. Install rsync.", file=sys.stderr)
                sys.exit(2)
            copy_with_progress_python(src, dst)
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
