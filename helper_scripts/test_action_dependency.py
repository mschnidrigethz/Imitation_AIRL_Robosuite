#!/usr/bin/env python3
"""Quick test: compare reward_net outputs on expert actions vs zero actions.

Usage:
  python helper_scripts/test_action_dependency.py --ckpt PATH --expert-npz PATH [--device cpu]
"""
import argparse
import numpy as np
import os
import sys

# Ensure project root is on sys.path so we can import scripts.* when running this file directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

# To avoid import issues when running as a standalone helper, embed the minimal
# functions we need here (copied/adapted from scripts/check_reward_net.py).
def make_franka_env():
  try:
    import robosuite as suite
    from robosuite.wrappers.gym_wrapper import GymWrapper
  except Exception:
    raise RuntimeError('robosuite not available in this environment')
  env = suite.make(
    env_name='Lift',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=20,
  )
  return GymWrapper(env)


def safe_torch_load(path, map_location='cpu', allow_unsafe=False):
  import torch
  try:
    return torch.load(path, map_location=map_location)
  except Exception:
    # best-effort fallback: try adding BasicShapedRewardNet to safe globals
    try:
      from imitation.rewards.reward_nets import BasicShapedRewardNet
      import torch.serialization as ts
      ts.add_safe_globals([BasicShapedRewardNet])
      return torch.load(path, map_location=map_location)
    except Exception:
      if allow_unsafe:
        return torch.load(path, map_location=map_location, weights_only=False)
      raise


def load_reward_model(path, obs_space, act_space, device='cpu', allow_unsafe=False):
  import torch
  print('Loading reward model...', path)
  ckpt = safe_torch_load(path, map_location=device, allow_unsafe=allow_unsafe)
  if isinstance(ckpt, dict):
    # try nested keys
    for k in ('state_dict', 'model', 'reward_net'):
      if k in ckpt and isinstance(ckpt[k], dict):
        ckpt = ckpt[k]
        break
    state = ckpt
    # build BasicShapedRewardNet with sensible defaults
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    net = BasicShapedRewardNet(
      observation_space=obs_space,
      action_space=act_space,
      reward_hid_sizes=(32,),
      potential_hid_sizes=(32, 32),
      use_state=True,
      use_action=True,
      use_next_state=False,
      use_done=False,
    )
    try:
      net.load_state_dict(state)
    except Exception:
      # best-effort: ignore if shapes mismatch; return net without state
      pass
    return net.to(device)
  if isinstance(ckpt, torch.nn.Module):
    return ckpt.to(device)
  raise RuntimeError('Unrecognized checkpoint format')


def rewards_from_net(net, obs, acts, device='cpu'):
  import torch
  import inspect
  net.eval()
  next_obs = None
  dones = None
  if isinstance(obs, (tuple, list)):
    if len(obs) >= 1:
      obs, next_obs, dones = obs[0], (obs[1] if len(obs) > 1 else None), (obs[2] if len(obs) > 2 else None)
  with torch.no_grad():
    o = torch.tensor(obs, dtype=torch.float32, device=device)
    a = torch.tensor(acts, dtype=torch.float32, device=device)
    next_o = None
    done_t = None
    if next_obs is not None:
      next_o = torch.tensor(next_obs, dtype=torch.float32, device=device)
    if dones is not None:
      done_t = torch.tensor(dones, dtype=torch.float32, device=device)

    sig = None
    try:
      sig = inspect.signature(net.forward)
    except Exception:
      try:
        sig = inspect.signature(net.__call__)
      except Exception:
        sig = None

    def call_net(o_t, a_t, next_o_t=None, done_t=None):
      try:
        if next_o_t is not None and done_t is not None:
          return net(o_t, a_t, next_o_t, done_t)
        if next_o_t is not None:
          return net(o_t, a_t, next_o_t)
        return net(o_t, a_t)
      except TypeError:
        kwargs = {}
        if 'next_state' in (sig.parameters if sig else {}):
          kwargs['next_state'] = next_o_t
        if 'done' in (sig.parameters if sig else {}):
          kwargs['done'] = done_t
        return net(o_t, a_t, **kwargs)

    try:
      r = call_net(o, a, next_o, done_t)
    except Exception:
      if hasattr(net, 'predict_reward'):
        if next_o is not None and done_t is not None:
          r = net.predict_reward(o, a, next_o, done_t)
        elif next_o is not None:
          r = net.predict_reward(o, a, next_o)
        else:
          r = net.predict_reward(o, a)
      else:
        raise

  r = torch.as_tensor(r).view(-1).cpu().numpy()
  return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='path to reward checkpoint (state_dict)')
    p.add_argument('--expert-npz', default='trajectories/merged_real_dataset_1.1to1.6_for_training.npz')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    data = np.load(args.expert_npz, allow_pickle=True)
    obs = data['observations']
    acts = data['actions']
    dones = data.get('dones', None)
    L = acts.shape[0]
    obs_t = obs[:L]
    obs_tp1 = obs[1:L+1] if obs.shape[0] >= L+1 else None

    env = make_franka_env()
    net = load_reward_model(args.ckpt, env.observation_space, env.action_space, device=args.device)

    r_expert = rewards_from_net(net, (obs_t, obs_tp1, dones), acts, device=args.device)
    r_zero = rewards_from_net(net, (obs_t, obs_tp1, dones), np.zeros_like(acts), device=args.device)

    print('Expert per-step: mean={:.6f} std={:.6f} sum={:.6f}'.format(r_expert.mean(), r_expert.std(), r_expert.sum()))
    print('Zero-action per-step: mean={:.6f} std={:.6f} sum={:.6f}'.format(r_zero.mean(), r_zero.std(), r_zero.sum()))
    print('Difference (expert - zero): {:.6f}'.format(r_expert.mean() - r_zero.mean()))


if __name__ == '__main__':
    main()
