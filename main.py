import h5py
import numpy as np

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.data.types import Trajectory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

#from scripts.converter import convert_h5_to_npz

# -----------------------------------------------------
# 1. Load Expert Data (HDF5 format)
# -----------------------------------------------------

def load_npz_trajectories(path):
    """Load trajectories stored in NPZ format."""
    trajectories = []
    data = np.load(path, allow_pickle=True)

    obs = data["observations"]
    acts = data["actions"]
    dones = data["dones"]

    start_idx = 0
    for i, done in enumerate(dones):
        if done:
            trajectories.append(
                Trajectory(
                    obs=obs[start_idx:i+1],
                    acts=acts[start_idx:i],
                    infos=None,
                    terminal=True,
                )
            )
            start_idx = i+1
    return trajectories


# -----------------------------------------------------
# 2. Create Robosuite Environment Wrapper
# -----------------------------------------------------

def make_franka_env():
    env = suite.make(
        env_name="Lift",  # Franka cube lifting task
        robots="Panda",   # Franka Panda robot
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )
    env.obs_keys = [
        "robot0_joint_pos",       # joint_pos
        "robot0_joint_vel",       # joint_vel
        "robot0_eef_pos",         # eef_pos
        "robot0_eef_quat",        # eef_quat
        "robot0_gripper_qpos",    # gripper_pos
        "cube_pos",               # cube_positions
        "cube_quat",              # cube_orientations
        "object-state"            # object
    ]
    return GymWrapper(env)


# Vectorized environment using DummyVecEnv
venv = DummyVecEnv([make_franka_env for _ in range(4)])


# -----------------------------------------------------
# 3. Expert Trajectories
# -----------------------------------------------------

expert_trajectories = load_npz_trajectories("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6.npz")


# -----------------------------------------------------
# 4. Define RL Policy & AIRL
# -----------------------------------------------------

policy = PPO("MlpPolicy", venv, verbose=1)

#reward_net = BasicRewardNet(
#    observation_space=venv.observation_space,
#    action_space=venv.action_space,
#    hidden_sizes=(64, 64),
#)

env = make_franka_env()
print("Generated observation shape:", env.observation_space.shape)


reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    reward_hid_sizes=(32,),  # Hidden layer sizes for the reward MLP
    potential_hid_sizes=(32, 32),  # Hidden layer sizes for the potential MLP
    use_state=True,
    use_action=True,
    use_next_state=False,
    use_done=False,
    discount_factor=0.99,
)

airl_trainer = airl.AIRL(
    demonstrations=expert_trajectories,
    venv=venv,
    gen_algo=policy,
    demo_batch_size=32,
    #new
    n_disc_updates_per_round=4,
    reward_net=reward_net,
    #expert_batch_size=32,
    #gen_batch_size=32,
)

# -----------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------
print("Expert observation shape:", expert_trajectories[0].obs.shape)
print("Generated observation shape:", env.observation_space.shape)
print("Starting AIRL training...")


#airl_trainer.train(n_epochs=50)
airl_trainer.train(100_000)

# Save trained policy
policy.save("airl_franka_cube_policy")
print("Training complete. Policy saved.")
