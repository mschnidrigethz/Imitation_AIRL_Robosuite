#import numpy as np
#
#data = np.load("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.npz")
##print(list(data.keys()))
#import numpy as np
#
#data = np.load("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.npz", allow_pickle=True)
#
#print("Keys in file:", data.keys())
#print("Observations shape:", data["observations"].shape)
#print("Actions shape:", data["actions"].shape)
#print("Dones shape:", data["dones"].shape)
#
## See a sample observation
#print("Sample observation vector:", data["observations"][0])
#print("Observation vector length:", len(data["observations"][0]))

import numpy as np

data = np.load("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.npz", allow_pickle=True)
print(data.files)  # lists stored arrays / keys

observations = data['observations']  # usually an array of dicts or structured arrays
print(type(observations))
print(len(observations))  # number of timesteps

# Peek at one observation
print(observations[0])