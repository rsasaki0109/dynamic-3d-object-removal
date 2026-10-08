#!/usr/bin/env python3
"""Accumulate a map using the exact manifest pose/deskew rules of the CLI."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dynamic_object_removal as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    if args.output.suffix != '.npy':
        parser.error('output must have .npy extension')
    scans, _ = core._load_scan_manifest(args.manifest)
    points = np.concatenate([scan for scan, _ in scans])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points)
    print(f'{len(scans)} scans, {len(points):,} points -> {args.output}')


if __name__ == '__main__':
    main()
