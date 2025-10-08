#!/usr/bin/env python3
"""Run a set of quick checks on a saved reward checkpoint:
 - inspect state_dict keys
 - compare expert vs zero-action rewards

Usage:
  python helper_scripts/run_quick_checks.py --ckpt PATH [--expert-npz PATH]
"""
import argparse
import subprocess
import sys
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--expert-npz', default='trajectories/merged_real_dataset_1.1to1.6_for_training.npz')
    args = p.parse_args()

    print('Inspecting checkpoint keys...')
    subprocess.check_call([sys.executable, '-u', os.path.join('helper_scripts', 'inspect_checkpoint_keys.py'), '--ckpt', args.ckpt])

    print('\nRunning action dependency test...')
    subprocess.check_call([sys.executable, '-u', os.path.join('helper_scripts', 'test_action_dependency.py'), '--ckpt', args.ckpt, '--expert-npz', args.expert_npz])


if __name__ == '__main__':
    main()
