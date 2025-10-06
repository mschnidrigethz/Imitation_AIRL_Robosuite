import h5py
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.hdf5'
with h5py.File(path, 'r') as f:
    g = f['data']['demo_0']['obs']
    cur = 0
    for k, item in g.items():
        per = int(np.prod(item.shape[1:])) if item.ndim > 1 else 1
        print(f"{k}: slice [{cur}:{cur+per}]  length={per}")
        cur += per
    print('final cur', cur)
