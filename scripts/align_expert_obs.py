"""align_expert_obs.py

Helpers to inspect expert HDF5/NPZ trajectories and to select/reorder
flattened observation vectors so they match a target observation layout
(for example the robosuite observation keys flattened order).

Usage examples (run locally where h5py and robosuite are installed):

Inspect an NPZ:
  python scripts/align_expert_obs.py --inspect-npz trajectories/merged_real_dataset_1.1to1.6.npz

Inspect an HDF5 to see dataset names and per-dataset flattened sizes (this
reveals the ordering used by the converter script which concatenates datasets
in recursion order):
  python scripts/align_expert_obs.py --inspect-hdf5 trajectories/merged_real_dataset_1.1to1.6.hdf5

Create a new NPZ keeping only a subset of fields (using an HDF5 to infer
the field-to-slice mapping):
  python scripts/align_expert_obs.py --align \
      --npz-in trajectories/merged_real_dataset_1.1to1.6.npz \
      --hdf5 trajectories/merged_real_dataset_1.1to1.6.hdf5 \
      --keep robot0_eef_pos,cube_pos,gripper_to_cube_pos \
      --npz-out trajectories/merged_real_dataset_1.1to1.6_aligned.npz

If you don't have an HDF5 but know index ranges you can provide --keep-indexes
with comma-separated ranges (e.g. 0:7,7:14,30:33) or single indices (e.g. 0,1,2).

Notes:
- This script does not assume robosuite is importable in the current environment
  where you edit the repo. To discover the target robosuite keys and sizes, run
  the helper function `print_robosuite_obs_spec()` on a machine that has robosuite
  installed (provided in the "--print-robosuite" command).
- The HDF5 -> NPZ converter in this repo flattened observations by recursing
  group.items() and concatenating ravel() of each dataset; this script uses the
  same recursion order to infer per-field slices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Sequence

import numpy as np


def traverse_hdf5_datasets(h5file) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return a list of (path, shape) for all datasets in the file using the
    same recursion order that the converter used (depth-first recursion over
    group.items()).
    """
    import h5py

    items: List[Tuple[str, Tuple[int, ...]]] = []

    def recurse(group, prefix=""):
        for key, item in group.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                items.append((path, tuple(item.shape)))
            else:
                recurse(item, path)

    recurse(h5file)
    return items


def safe_concat(arrs: List[np.ndarray], axis: int = 0, names: List[str] = None) -> np.ndarray:
    """Concatenate arrays but check that all non-concatenation dims match.

    Raises a ValueError with a helpful message if sizes mismatch.
    """
    if not arrs:
        return np.concatenate(arrs, axis=axis)

    # Normalize arrays to numpy
    arrs = [np.asarray(a) for a in arrs]

    # Determine ranks
    nd = arrs[0].ndim
    for a in arrs[1:]:
        if a.ndim != nd:
            raise ValueError(f'All arrays must have same ndim; found {arrs[0].ndim} and {a.ndim}')

    # Check all dims except concat axis
    for dim in range(nd):
        if dim == axis:
            continue
        sizes = [a.shape[dim] for a in arrs]
        if any(s != sizes[0] for s in sizes):
            # Build helpful message
            msg = f"Dimension mismatch on axis {dim}: sizes={sizes}."
            if names:
                msg += f" Arrays/fields: {names}"
            raise ValueError(msg)

    return np.concatenate(arrs, axis=axis)


def compute_flat_lengths(datasets: Sequence[Tuple[str, Tuple[int, ...]]]) -> List[Tuple[str, int]]:
    """Convert dataset shapes to flattened lengths. For datasets that are
    time-major (first dim timesteps), this returns the per-timestep length if
    possible (i.e., shape[1:] -> ravel length). The converter stores per
    timestep observations by raveling dataset[t] for each dataset in the
    recursion order, so to reconstruct how the flattened obs vector was built
    we want the flattened size per timestep for each dataset.
    """
    lengths = []
    for path, shape in datasets:
        if len(shape) == 0:
            # scalar
            per_t = 1
        elif len(shape) == 1:
            # either timesteps or a 1D vector (ambiguous). We assume timesteps
            # were concatenated per-timestep so treat shape[0] as timesteps and
            # per-timestep length 1. The converter logic reads item[t] and
            # ravels it, so item[t] must be indexable.
            per_t = 1
        else:
            # shape like (timesteps, dim1, dim2, ...)
            per_t = int(np.prod(shape[1:]))
        lengths.append((path, per_t))
    return lengths


def build_slices(lengths: Sequence[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
    """Given list of (name, per_timestep_len) return list of (name, start, end)
    with cumulative offsets (end exclusive)."""
    slices: List[Tuple[str, int, int]] = []
    cur = 0
    for name, l in lengths:
        slices.append((name, cur, cur + l))
        cur += l
    return slices


def inspect_npz(path: str) -> None:
    data = np.load(path, allow_pickle=True)
    print("Keys in NPZ:", data.files)
    if 'observations' in data:
        obs = data['observations']
        print('observations type:', type(obs))
        try:
            print('observations shape:', obs.shape)
        except Exception:
            pass
        if obs.size:
            first = obs.reshape(-1, obs.shape[-1])[0] if obs.ndim == 2 else obs[0]
            print('one obs length (ravel):', np.ravel(first).shape)


def inspect_hdf5(path: str) -> None:
    try:
        import h5py
    except ImportError:
        print('h5py is required to inspect HDF5 files. Please install h5py in your environment.')
        sys.exit(2)

    with h5py.File(path, 'r') as f:
        # find all dataset paths and shapes
        datasets = traverse_hdf5_datasets(f)
        print('Found datasets:')
        for p, sh in datasets:
            print(f'  {p} -> shape={sh}')

        lengths = compute_flat_lengths(datasets)
        slices = build_slices(lengths)
        print('\nPer-timestep flattened lengths and cumulative slices:')
        for name, start, end in slices:
            print(f'  {name}: {end-start}  slice [{start}:{end}]')


def print_robosuite_obs_spec() -> None:
    """Create the robosuite env and print its observation dict keys and per-key sizes.
    Requires robosuite installed in the current environment.
    """
    try:
        import robosuite as suite
    except Exception as e:
        print('Could not import robosuite:', e)
        print('Run this on a machine/environment with robosuite installed to print the target obs spec.')
        return

    from robosuite.wrappers.gym_wrapper import GymWrapper
    env = suite.make(env_name='Lift', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     use_object_obs=True)
    obs = env.reset()
    print('Robosuite reset() returned type:', type(obs))
    total = 0
    if isinstance(obs, dict):
        print('Robosuite observation keys and sizes:')
        for k, v in obs.items():
            v = np.array(v)
            print(f'  {k}: {v.shape}  (flattened {v.ravel().shape[0]})')
            total += v.ravel().shape[0]
        print('Total flattened robosuite obs length:', total)
    else:
        print('Non-dict observation, flattened length:', np.ravel(obs).shape[0])


def align_npz_using_slices(npz_in: str, npz_out: str, slices: List[Tuple[str, int, int]], keep: List[str]) -> None:
    """Given an input NPZ with 2D observations (N, L), and a slices list of
    (name,start,end) that describes how the flattened observation was built,
    produce a new NPZ that keeps only the requested `keep` fields in the order
    provided.
    """
    data = np.load(npz_in, allow_pickle=True)
    if 'observations' not in data:
        raise ValueError('NPZ does not contain observations key')
    obs = data['observations']
    if obs.ndim != 2:
        raise ValueError('This script expects observations to be a 2D array (timesteps, flat_dim)')

    name_to_slice = {name: (s, e) for name, s, e in slices}
    out_cols = []
    col_names = []
    for k in keep:
        if k not in name_to_slice:
            raise KeyError(f'Field {k} not found in slices mapping')
        s, e = name_to_slice[k]
        out_cols.append(obs[:, s:e])
        col_names.append(k)

    new_obs = safe_concat(out_cols, axis=1, names=col_names)
    print('Old obs shape:', obs.shape, 'New obs shape:', new_obs.shape)

    # keep actions and dones if present
    out_dict = {'observations': new_obs}
    for key in ['actions', 'dones']:
        if key in data:
            out_dict[key] = data[key]

    np.savez_compressed(npz_out, **out_dict)
    print('Saved aligned NPZ to', npz_out)


def parse_keep_indexes(txt: str) -> List[Tuple[int, int]]:
    """Parse a string like '0:7,7:14,30:33' or '0,1,2' -> list of (s,e) slices
    where single ints are treated as 1-length ranges."""
    parts = [p.strip() for p in txt.split(',') if p.strip()]
    ranges: List[Tuple[int, int]] = []
    for p in parts:
        if ':' in p:
            s, e = p.split(':')
            ranges.append((int(s), int(e)))
        else:
            i = int(p)
            ranges.append((i, i+1))
    return ranges


def main():
    parser = argparse.ArgumentParser(description='Align expert observations to a target subset/order')
    parser.add_argument('--inspect-npz', help='Print basic info about an NPZ file')
    parser.add_argument('--inspect-hdf5', help='Print HDF5 datasets and per-timestep flattened sizes')
    parser.add_argument('--print-robosuite', action='store_true', help='Instantiate robosuite Lift env and print obs keys and sizes (needs robosuite installed)')

    parser.add_argument('--align', action='store_true', help='Run alignment from NPZ using HDF5-inferred slices')
    parser.add_argument('--npz-in', help='Input npz with observations (2D)')
    parser.add_argument('--hdf5', help='HDF5 used to infer the flattening order')
    parser.add_argument('--npz-out', help='Output aligned npz path')
    parser.add_argument('--to-robosuite', action='store_true', help='Create a robosuite-like (53-dim) observation NPZ using HDF5 slices')
    parser.add_argument('--keep', help='Comma-separated list of dataset paths to keep, in order (as printed by --inspect-hdf5)')
    parser.add_argument('--keep-indexes', help='Comma-separated list of index ranges to keep if you want to specify slices directly (e.g. 0:7,7:14)')

    args = parser.parse_args()

    if getattr(args, 'inspect_npz', None):
        inspect_npz(args.inspect_npz)
        return

    if args.inspect_hdf5:
        inspect_hdf5(args.inspect_hdf5)
        return

    if args.print_robosuite:
        print_robosuite_obs_spec()
        return

    if args.align:
        if not args.npz_in or not args.npz_out:
            print('For --align you must provide --npz-in and --npz-out')
            sys.exit(2)
        if args.hdf5 is None and args.keep_indexes is None:
            print('For automatic mapping you must provide --hdf5 that was used to create the npz, or provide --keep-indexes')
            sys.exit(2)

        slices = []
        if args.hdf5:
            try:
                import h5py
            except ImportError:
                print('h5py is required to read the HDF5 file. Install it to use --hdf5')
                sys.exit(2)
            with h5py.File(args.hdf5, 'r') as f:
                datasets = traverse_hdf5_datasets(f)
                lengths = compute_flat_lengths(datasets)
                slices = build_slices(lengths)

        if args.keep_indexes:
            ranges = parse_keep_indexes(args.keep_indexes)
            # convert ranges to names if we have slices mapping
            if slices:
                name_slices = []
                for (name, s, e) in slices:
                    name_slices.append((name, s, e))
                # Only use names provided explicitly via --keep
                if args.keep:
                    keep = [k.strip() for k in args.keep.split(',') if k.strip()]
                    align_npz_using_slices(args.npz_in, args.npz_out, name_slices, keep)
                    return
                else:
                    # apply index ranges directly to input columns
                    data = np.load(args.npz_in, allow_pickle=True)
                    obs = data['observations']
                    cols = [obs[:, s:e] for (s, e) in ranges]
                    new_obs = safe_concat(cols, axis=1)
                    out = {'observations': new_obs}
                    for key in ['actions', 'dones']:
                        if key in data:
                            out[key] = data[key]
                    np.savez_compressed(args.npz_out, **out)
                    print('Saved', args.npz_out)
                    return

        if args.keep:
            keep = [k.strip() for k in args.keep.split(',') if k.strip()]
            if not slices:
                raise RuntimeError('No slices mapping available to resolve names. Provide --hdf5.')
            align_npz_using_slices(args.npz_in, args.npz_out, slices, keep)
            return

    if getattr(args, 'to_robosuite', False):
        # Build a robosuite-like 53-dim observation (robot0_proprio-state (43) + object-state (10))
        if not args.npz_in or not args.npz_out or not args.hdf5:
            print('For --to-robosuite provide --npz-in, --npz-out and --hdf5')
            sys.exit(2)
        # build mapping from hdf5
        try:
            import h5py
        except ImportError:
            print('h5py required to read HDF5 file for mapping')
            sys.exit(2)
        with h5py.File(args.hdf5, 'r') as f:
            # read obs group (demo_0 assumed to reflect structure)
            obs_group = f['data']['demo_0']['obs']
            # create a name->(s,e) mapping using same recursion order used earlier
            datasets = traverse_hdf5_datasets(obs_group)
            lengths = compute_flat_lengths(datasets)
            slices = build_slices(lengths)

        name_to_slice = {name: (s, e) for name, s, e in slices}

        data = np.load(args.npz_in, allow_pickle=True)
        obs = data['observations']
        if obs.ndim != 2:
            raise RuntimeError('expected 2D observations')

        # Helper to grab columns by field name
        def cols_for(name):
            if name not in name_to_slice:
                raise KeyError(name)
            s, e = name_to_slice[name]
            return obs[:, s:e]

        # Compose robot0_proprio-state (43): we'll approximate the fields
        # using available expert fields:
        # joint_pos (7), joint_pos_cos (7)=cos(joint_pos), joint_pos_sin (7)=sin(joint_pos),
        # joint_vel (7), eef_pos (3), eef_quat (4), eef_quat_site (4 we duplicate eef_quat),
        # robot0_gripper_qpos (2) approximated from gripper_pos[:2] if available, else zeros,
        # robot0_gripper_qvel (2) zeros.
        jp = cols_for('joint_pos')
        jv = cols_for('joint_vel')
        eef_pos = cols_for('eef_pos')
        eef_quat = cols_for('eef_quat')
        gripper_pos = cols_for('gripper_pos')

        jp_cos = np.cos(jp)
        jp_sin = np.sin(jp)

        # eef_quat_site: duplicate eef_quat as approximation
        eef_quat_site = eef_quat.copy()

        # gripper qpos approximation: take first 2 dims of gripper_pos if >=2, else pad
        if gripper_pos.shape[1] >= 2:
            gripper_qpos = gripper_pos[:, :2]
        else:
            gripper_qpos = np.zeros((gripper_pos.shape[0], 2), dtype=gripper_pos.dtype)

        gripper_qvel = np.zeros_like(gripper_qpos)
        
        # Approximation of gripper qvel via finite differences (assuming 50Hz sampling)
        #dt = 1.0 / 50.0     # falls Samplerate 50 Hz; passe an
        ## forward difference, pad first row with zeros or repeat
        #gripper_qvel = np.vstack([np.zeros((1, gripper_qpos.shape[1])), np.diff(gripper_qpos, axis=0)]) / dt

        proprio = np.concatenate([jp, jp_cos, jp_sin, jv, eef_pos, eef_quat, eef_quat_site, gripper_qpos, gripper_qvel], axis=1)
        assert proprio.shape[1] == 43, f'proprio width {proprio.shape[1]} != 43'

        # Compose object-state (10): cube_pos (3), cube_quat (4), gripper_to_cube_pos (3) = eef_pos - cube_pos
        cube_pos = cols_for('cube_positions')
        cube_quat = cols_for('cube_orientations')
        gripper_to_cube = eef_pos - cube_pos
        obj = np.concatenate([cube_pos, cube_quat, gripper_to_cube], axis=1)
        assert obj.shape[1] == 10, f'object width {obj.shape[1]} != 10'

        new_obs = np.concatenate([proprio, obj], axis=1)
        print('Old obs shape:', obs.shape, 'New obs shape:', new_obs.shape)

        out = {'observations': new_obs}
        for key in ['actions', 'dones']:
            if key in data:
                out[key] = data[key]
        np.savez_compressed(args.npz_out, **out)
        print('Saved robosuite-like npz to', args.npz_out)
        return

        print('Nothing to do. See --help for usage.')


if __name__ == '__main__':
    main()
