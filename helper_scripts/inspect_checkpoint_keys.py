#!/usr/bin/env python3
"""Inspect state_dict keys inside a checkpoint file and print keys related to normalization or running stats."""
import argparse
import torch
import os
import sys

# ensure repo root on path when run directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    args = p.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ck, dict):
        for k in ('state_dict', 'model', 'reward_net'):
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
    if not isinstance(ck, dict):
        print('Checkpoint not a dict; type', type(ck))
        return
    keys = sorted(list(ck.keys()))
    print('Total keys:', len(keys))
    for k in keys:
        if 'running' in k.lower() or 'norm' in k.lower() or 'running_mean' in k or 'running_var' in k:
            print('NORM/STAT KEY:', k)
    # print some example keys
    print('\nSome keys (first 200):')
    for k in keys[:200]:
        print(k)


if __name__ == '__main__':
    main()
