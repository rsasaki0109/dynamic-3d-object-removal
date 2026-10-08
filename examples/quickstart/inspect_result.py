#!/usr/bin/env python3
"""Check that cleaned NPY points match the saved boolean keep mask."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-map', required=True)
    parser.add_argument('--cleaned', required=True)
    parser.add_argument('--mask', required=True)
    args = parser.parse_args()
    raw = np.load(args.input_map)
    cleaned = np.load(args.cleaned)
    mask = np.load(args.mask)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        parser.error('input map must be finite N x 3 XYZ')
    if mask.dtype != np.bool_ or mask.shape != (len(raw),):
        parser.error('mask must be boolean with one entry per input point')
    if not np.array_equal(raw[mask], cleaned):
        parser.error('cleaned points differ from input_map[mask]')
    print(f'Input: {len(raw):,}; kept: {mask.sum():,}; removed: {(~mask).sum():,}')
    print('Saved points and mask agree. Removal counts are not accuracy metrics.')


if __name__ == '__main__':
    main()
