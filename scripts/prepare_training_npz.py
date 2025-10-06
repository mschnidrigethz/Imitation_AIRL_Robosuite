#!/usr/bin/env python3
"""Prepare an expert NPZ so it exactly matches a Robosuite env's observation
and action shapes.

This script will:
- instantiate the Robosuite env (Lift/Panda) to learn the target obs/action sizes
- if needed, call the aligner to produce a robosuite-like observations NPZ
- resize/truncate/pad actions to match the env.action_space
- save a new NPZ suitable for training

Run inside the conda env that has robosuite installed (env_imitation):
  conda run -n env_imitation python3 scripts/prepare_training_npz.py
"""

import argparse
import os
import subprocess
import sys
import numpy as np


def get_env_spec():
    try:
        import robosuite as suite
        from robosuite.wrappers.gym_wrapper import GymWrapper
    except Exception as e:
        print('Failed to import robosuite in this environment:', e)
        sys.exit(2)
    env = suite.make(env_name='Lift', robots='Panda', has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True)
    wrapped = GymWrapper(env)
    obs_dim = int(np.prod(wrapped.observation_space.shape))
    # action space may be Box with shape attribute
    try:
        action_dim = int(wrapped.action_space.shape[0])
    except Exception:
        # fallback: sample an action and use its length
        action_dim = len(wrapped.action_space.sample())
    return obs_dim, action_dim


def run_align(npz_in, hdf5, npz_out):
    # Call the align_expert_obs script to produce a robosuite-style NPZ.
    cmd = [sys.executable, 'scripts/align_expert_obs.py', '--to-robosuite', '--npz-in', npz_in, '--hdf5', hdf5, '--npz-out', npz_out]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def prepare(npz_in, hdf5, npz_out):
    # Determine env spec
    obs_dim, action_dim = get_env_spec()
    print('Target env obs_dim=', obs_dim, 'action_dim=', action_dim)

    # Load input NPZ
    data = np.load(npz_in, allow_pickle=True)
    obs = data['observations']
    acts = data['actions'] if 'actions' in data else None
    dones = data['dones'] if 'dones' in data else None

    # If obs not matching, try to create robosuite-aligned NPZ using hdf5 mapping
    if obs.ndim == 2 and obs.shape[1] != obs_dim:
        if hdf5 is None:
            raise RuntimeError(f'Observations have shape {obs.shape[1]} but target is {obs_dim}. Provide --hdf5 to allow automatic alignment.')
        tmp = npz_in + '.robosuite_tmp.npz'
        run_align(npz_in, hdf5, tmp)
        data = np.load(tmp, allow_pickle=True)
        obs = data['observations']
        acts = data['actions'] if 'actions' in data else acts
        dones = data['dones'] if 'dones' in data else dones
        os.remove(tmp)

    if obs.ndim != 2 or obs.shape[1] != obs_dim:
        raise RuntimeError(f'After alignment, observations have shape {obs.shape}. Expected second-dim {obs_dim}.')

    # Adjust actions to match action_dim
    if acts is None:
        print('No actions found in input NPZ; creating zeros actions of shape (N-1, action_dim)')
        acts = np.zeros((obs.shape[0]-1, action_dim), dtype=np.float32)
    else:
        N, a_w = acts.shape
        if a_w == action_dim:
            print('Actions already match action_dim')
        elif a_w > action_dim:
            print(f'Truncating actions from width {a_w} to {action_dim} (dropping tail columns)')
            acts = acts[:, :action_dim]
        else:
            print(f'Padding actions from width {a_w} to {action_dim} with zeros')
            pad = np.zeros((acts.shape[0], action_dim - a_w), dtype=acts.dtype)
            acts = np.concatenate([acts, pad], axis=1)

    # Ensure dones length matches acts length
    if dones is None:
        dones = np.zeros((acts.shape[0],), dtype=np.bool_)
    elif len(dones) != acts.shape[0]:
        # If dones corresponded to original obs length, adjust: our conventions assume dones length == acts length
        if len(dones) == obs.shape[0]:
            dones = dones[:-1]
        else:
            # truncate or pad
            if len(dones) > acts.shape[0]:
                dones = dones[:acts.shape[0]]
            else:
                pad = np.array([False] * (acts.shape[0] - len(dones)), dtype=np.bool_)
                dones = np.concatenate([dones, pad], axis=0)

    print('Final shapes: observations', obs.shape, 'actions', acts.shape, 'dones', dones.shape)

    np.savez_compressed(npz_out, observations=obs, actions=acts, dones=dones)
    print('Saved prepared NPZ to', npz_out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz-in', default='trajectories/merged_real_dataset_1.1to1.6.npz')
    p.add_argument('--hdf5', default='trajectories/merged_real_dataset_1.1to1.6.hdf5')
    p.add_argument('--npz-out', default='trajectories/merged_real_dataset_1.1to1.6_for_training.npz')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare(args.npz_in, args.hdf5, args.npz_out)
