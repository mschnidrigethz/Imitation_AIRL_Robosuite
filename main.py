import h5py
import numpy as np
import os
import time
import torch

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
    return GymWrapper(env)


# Vectorized environment using DummyVecEnv
venv = DummyVecEnv([make_franka_env for _ in range(4)])


# -----------------------------------------------------
# 3. Expert Trajectories
# -----------------------------------------------------

expert_trajectories = load_npz_trajectories("/home/chris/Imitation/trajectories/merged_real_dataset_1.1to1.6_for_training.npz")


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
# Quick validation: ensure expert obs dim matches environment obs dim
if len(expert_trajectories) == 0:
    raise RuntimeError('No expert trajectories loaded; check NPZ path')

# Flatten env observation size
env_obs_dim = int(np.prod(env.observation_space.shape))
expert_obs_dim = int(np.ravel(expert_trajectories[0].obs[0]).shape[0])
print(f"Expert obs dim: {expert_obs_dim}; Env obs dim: {env_obs_dim}")
if expert_obs_dim != env_obs_dim:
    raise ValueError(
        f"Expert observation dimension ({expert_obs_dim}) does not match environment observation dimension ({env_obs_dim}).\n"
        "If you intended to use the robosuite-53 representation, create an aligned dataset first using the helper script:\n"
        "  conda run -n env_imitation python3 scripts/align_expert_obs.py --to-robosuite --npz-in trajectories/merged_real_dataset_1.1to1.6.npz --hdf5 trajectories/merged_real_dataset_1.1to1.6.hdf5 --npz-out trajectories/merged_real_dataset_1.1to1.6_robosuite.npz\n"
    )


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
#print("Expert observation shape:", expert_trajectories[0].obs.shape)
#print("Generated observation shape:", env.observation_space.shape)
#print("Starting AIRL training...")


#airl_trainer.train(n_epochs=50)
airl_trainer.train(2_000_000)

# Save trained policy and reward net to a timestamped folder so we don't lose artifacts
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("output", "manual_runs", f"franka_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

# policy
policy_path = os.path.join(out_dir, "gen_policy")
policy.save(policy_path)

# reward net: save state_dict and full object where possible
try:
    reward_state_path = os.path.join(out_dir, "reward_net_state.pth")
    torch.save(reward_net.state_dict(), reward_state_path)
    # also try to save the whole object (may fail if there are lambdas or non-picklable members)
    reward_full_path = os.path.join(out_dir, "reward_net_full.pth")
    torch.save(reward_net, reward_full_path)
    print(f"Saved reward net state to {reward_state_path} and full object to {reward_full_path}")
except Exception as e:
    print("Warning: saving full reward_net object failed:", e)
    print(f"Reward state was saved to {reward_state_path} if available.")

print(f"Training complete. Artifacts saved under {out_dir}")
