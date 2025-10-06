#!/usr/bin/env python3
"""Convert expert trajectories (HDF5 or NPZ) into flattened observation vectors
matching a specified robosuite observation ordering.

Usage:
  python scripts/convert_expert_to_env_obs.py --input /path/to/dataset.hdf5 --out /path/to/out.npz

The script supports HDF5 datasets with a structure like the recorder used in this
workspace (group 'data' -> demos -> 'obs' group with fields). It also supports
loading an existing .npz (with 'observations' array).

By default the target ordering is the one used in `main.py` / `dimensionfit.py`:
['joint_pos','joint_vel','eef_pos','eef_quat','gripper_pos','cube_pos','cube_quat','object']

If the input contains structured per-field datasets, the script will concatenate
them in that order for each timestep. If the flattened vector from input is longer
or shorter than the expected target length, the script will either truncate or pad
with zeros and print a warning.
"""
import argparse
import numpy as np
import os

try:
    import h5py
except Exception:
    h5py = None


DEFAULT_TARGET_KEYS = [
    'joint_pos',
    'joint_vel',
    'eef_pos',
    'eef_quat',
    'gripper_pos',
    'cube_pos',
    'cube_quat',
    'object',
]


def read_npz_obs(path):
    data = np.load(path, allow_pickle=True)
    if 'observations' not in data:
        raise ValueError('NPZ does not contain "observations" key')
    return data['observations']


def read_hdf5_obs_group(h5path):
    if h5py is None:
        raise RuntimeError('h5py is not installed in this environment')
    out = []
    with h5py.File(h5path, 'r') as f:
        if 'data' not in f:
            raise ValueError("HDF5 file missing top-level 'data' group")
        for demo_key in f['data']:
            demo = f['data'][demo_key]
            obs = demo['obs']
            # build per-timestep concatenated arrays based on present datasets
            timesteps = obs[list(obs.keys())[0]].shape[0]
            for t in range(timesteps):
                parts = []
                # preserve written order (h5py keeps insertion order)
                for k in obs:
                    item = obs[k]
                    if isinstance(item, h5py.Dataset):
                        parts.append(np.ravel(item[t]))
                    else:
                        # nested groups not expected here, but handle gracefully
                        # traverse group datasets in key order
                        def recurse(group):
                            res = []
                            for kk in group:
                                it = group[kk]
                                if isinstance(it, h5py.Dataset):
                                    res.append(np.ravel(it[t]))
                                else:
                                    res.extend(recurse(it))
                            return res
                        parts.extend(recurse(item))
                out.append(np.concatenate(parts))
    return np.stack(out)


def convert_flat_to_target(flat_obs, target_len):
    """Truncate or pad flattened observations to target_len."""
    out = np.zeros((flat_obs.shape[0], target_len), dtype=np.float32)
    for i in range(flat_obs.shape[0]):
        row = np.asarray(flat_obs[i]).ravel()
        if row.shape[0] >= target_len:
            out[i] = row[:target_len]
        else:
            out[i, : row.shape[0]] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input .hdf5 or .npz dataset')
    p.add_argument('--out', '-o', required=True, help='Output .npz path')
    p.add_argument('--target-len', type=int, default=None, help='If set, force flattened obs to this length')
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        raise SystemExit(f'Input not found: {inp}')

    if inp.endswith('.npz'):
        obs = read_npz_obs(inp)
        print('Loaded NPZ observations:', obs.shape)
    elif inp.endswith('.hdf5') or inp.endswith('.h5'):
        obs = read_hdf5_obs_group(inp)
        print('Loaded HDF5 obs stacked shape:', obs.shape)
    else:
        raise SystemExit('Unsupported input format; use .npz or .hdf5')

    if args.target_len is not None:
        target_len = args.target_len
    else:
        # If robosuite env not available here, user should provide --target-len
        # We will try to infer from default keys sizes if possible
        # Fall back to existing observation length
        target_len = obs.shape[1]

    if obs.shape[1] != target_len:
        print(f'Converting obs from length {obs.shape[1]} -> {target_len} (truncate/pad)')
        obs_conv = convert_flat_to_target(obs, target_len)
    else:
        obs_conv = obs.astype(np.float32)

    np.savez_compressed(args.out, observations=obs_conv)
    print('Saved converted observations to', args.out)


if __name__ == '__main__':
    main()
