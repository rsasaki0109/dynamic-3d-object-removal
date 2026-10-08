#!/usr/bin/env python3
"""Create a tiny synthetic CLI input; no downloads or real LiDAR claims."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    args.output.mkdir(parents=True)
    frames = []
    # A tall accumulated column and later low returns exercise both channels.
    for i in range(4):
        points = ([[5, 0, 0], [5, 0, 10], [5, 0, 1]] if i == 0
                  else [[7, 0, 0], [7, 0, 1.4]])
        np.save(args.output / f'scan{i}.npy', np.array(points, dtype=float))
        frames.append({'cloud': f'scan{i}.npy', 'pose': {
            'translation': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}})
    manifest = {'sensor_profile': {'name': 'synthetic example', 'deskewed': True},
                'frames': frames}
    (args.output / 'scans.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output / 'scans.json')


if __name__ == '__main__':
    main()
