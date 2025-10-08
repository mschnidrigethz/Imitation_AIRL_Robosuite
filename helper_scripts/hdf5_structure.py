import h5py

path = "/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.hdf5"

with h5py.File(path, "r") as f:
    def print_structure(name, obj):
        print(name, "->", type(obj))
    f.visititems(print_structure)
