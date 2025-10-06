#import h5py
#import numpy as np
import sys
#import os
#
#def convert_h5_to_npz(h5_path, npz_path):
#    """Convert HDF5 dataset to .npz."""
#    with h5py.File(h5_path, "r") as h5_file:
#        data_dict = {}
#
#        def recursively_extract(name, obj):
#            if isinstance(obj, h5py.Dataset):
#                data_dict[name] = obj[()]
#            elif isinstance(obj, h5py.Group):
#                for key, val in obj.items():
#                    recursively_extract(f"{name}/{key}" if name else key, val)
#
#        recursively_extract("", h5_file)
#    np.savez_compressed(npz_path, **data_dict)
#    print(f"[✔] Converted '{h5_path}' → '{npz_path}'")
#    
#if __name__ == "__main__":
#    
#    if len(sys.argv) != 3:
#        print("Usage: python hdf5TOnpz_converter.py <input_h5_path> <output_npz_path>")
#        sys.exit(1)
#    h5_path = sys.argv[1]
#    npz_path = sys.argv[2]
#
#    convert_h5_to_npz(h5_path, npz_path)

import h5py
import numpy as np

def extract_obs(group):
    """Flatten observation group into a single observation vector."""
    obs_list = []

    def recurse(group):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                obs_list.append(np.array(item))
            elif isinstance(item, h5py.Group):
                recurse(item)

    recurse(group)
    return np.concatenate([np.ravel(o) for o in obs_list], axis=0)


def convert_hdf5_to_npz(hdf5_path, npz_path):
    observations = []
    actions = []
    dones = []

    with h5py.File(hdf5_path, "r") as f:
        for demo_key in f["data"]:
            demo_group = f["data"][demo_key]

            obs = extract_obs(demo_group["obs"])
            act = np.array(demo_group["actions"])

            # obs is per timestep, but extract_obs returns all timesteps concatenated
            # We must split obs by timestep length
            timesteps = act.shape[0]

            # Re-extract observations timestep-by-timestep
            obs_per_timestep = []
            for t in range(timesteps):
                timestep_obs = []
                def recurse_t(group):
                    for key, item in group.items():
                        if isinstance(item, h5py.Dataset):
                            timestep_obs.append(np.ravel(item[t]))
                        elif isinstance(item, h5py.Group):
                            recurse_t(item)
                recurse_t(demo_group["obs"])
                obs_per_timestep.append(np.concatenate(timestep_obs))

            stacked = np.stack(obs_per_timestep)
            print(f"Trajectory {demo_key}: obs {stacked.shape}, actions {np.array(act[:-1]).shape}")
            observations.append(stacked)
            actions.append(act[:-1])

            dones.append(np.array([False] * (timesteps - 2) + [True]))

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    dones = np.concatenate(dones, axis=0)
    
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Dones shape: {dones.shape}")
    
    


    np.savez_compressed(npz_path,
                        observations=observations,
                        actions=actions,
                        dones=dones)
    print(f"Saved converted dataset to {npz_path}")


if __name__ == "__main__":
    hdf5_path = sys.argv[1]
    npz_path = sys.argv[2]
    convert_hdf5_to_npz(hdf5_path, npz_path)
