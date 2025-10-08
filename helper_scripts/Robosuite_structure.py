#import robosuite as suite
#from robosuite.wrappers.gym_wrapper import GymWrapper
#import numpy as np
#
#env = suite.make(
#    env_name="Lift",
#    robots="Panda",
#    has_renderer=False,
#    has_offscreen_renderer=False,
#    use_camera_obs=False,
#    use_object_obs=True,
#    control_freq=20,
#)
#
#wrapped_env = GymWrapper(env)
#
#obs, info = wrapped_env.reset()
#print("Robosuite observation shape:", np.array(obs).shape)

#import robosuite as suite
#import numpy as np
#
#env = suite.make(
#    env_name="Lift",
#    robots="Panda",
#    has_renderer=False,
#    has_offscreen_renderer=False,
#    use_camera_obs=False,
#    use_object_obs=True,
#)
#
#obs = env.reset()
#
#print("Observation type:", type(obs))
#print("Observation keys:", obs.keys() if isinstance(obs, dict) else "Not a dict")
#
#for k, v in obs.items():
#    print(f"{k}: {np.array(v).shape}")
#
#obs_sample = np.load("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.npz")["observations"][0]
#print(obs_sample.shape)

from itertools import combinations

obs_keys = {
    "robot0_joint_pos": 7,
    "robot0_joint_pos_cos": 7,
    "robot0_joint_pos_sin": 7,
    "robot0_joint_vel": 7,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_eef_quat_site": 4,
    "robot0_gripper_qpos": 2,
    "robot0_gripper_qvel": 2,
    "cube_pos": 3,
    "cube_quat": 4,
    "gripper_to_cube_pos": 3,
    "robot0_proprio-state": 43,
    "object-state": 10
}

target_length = 82

for r in range(1, len(obs_keys) + 1):
    for combo in combinations(obs_keys.keys(), r):
        if sum(obs_keys[k] for k in combo) == target_length:
            print("Matching keys:", combo, "→ length:", target_length)
