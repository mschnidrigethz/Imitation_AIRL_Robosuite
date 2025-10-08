import h5py
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.hdf5'
with h5py.File(path,'r') as f:
    data = f['data']
    demos = sorted(list(data.keys()))
    widths = {}
    for demo in demos:
        g = data[demo]
        if 'obs' not in g:
            print(demo, 'no obs group')
            continue
        obs_group = g['obs']
        per_fields = []
        total = 0
        for key, item in obs_group.items():
            if isinstance(item, h5py.Dataset):
                shape = item.shape
                per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                per_fields.append((key, per, shape))
                total += per
        widths[demo] = total
        print(f"{demo}: num_samples={g.attrs.get('num_samples', 'N/A')}, obs_width={total}")
        for name, per, shape in per_fields:
            print(f"  {name}: per={per}, shape={shape}")

    print('\nSummary:')
    uniq = {}
    for k,v in widths.items():
        uniq.setdefault(v, []).append(k)
    for w,dlist in uniq.items():
        print(w, '->', dlist)
