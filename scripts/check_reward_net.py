"""
python scripts/check_reward_net.py --reward output/manual_runs/franka_20251008_123837/reward_net_state.pth --expert-npz trajectories/merged_real_dataset_1.1to1.6_for_training.npz

Zu testendes Reward Netzt einfügen, dann wird es auf
Experten-Daten und random Daten evaluiert."""


import argparse
import numpy as np
import torch
import os
import time

try:
    import robosuite as suite
    from robosuite.wrappers.gym_wrapper import GymWrapper
except Exception:
    suite = None

try:
    import gym
    from gym import spaces
except Exception:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception:
        gym = None
        spaces = None


def make_franka_env():
    if suite is None:
        raise RuntimeError("robosuite not available in this environment")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
    )
    return GymWrapper(env)


def build_reward_net_from_spaces(obs_space, act_space):
    from imitation.rewards.reward_nets import BasicShapedRewardNet

    net = BasicShapedRewardNet(
        observation_space=obs_space,
        action_space=act_space,
        reward_hid_sizes=(32,),
        potential_hid_sizes=(),
        use_state=True,
        use_action=True,
        use_next_state=False,
        use_done=False,
        discount_factor=0.99,
    )
    return net


def safe_torch_load(path, map_location='cpu', allow_unsafe=False):
    # Try normal load first; if it fails due to safe globals, try to whitelist
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        # attempt safe whitelist if imitation class is referenced
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
    print("Loading reward model...", path)
    ckpt = safe_torch_load(path, map_location=device, allow_unsafe=allow_unsafe)

    # If ckpt is a state dict
    if isinstance(ckpt, dict):
        # ckpt might be nested: find the actual state_dict
        candidates = [ckpt]
        for key in ('state_dict', 'model', 'reward_net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                candidates.append(ckpt[key])

        state = None
        for c in candidates:
            # pick the candidate that contains parameter tensors
            if isinstance(c, dict) and any(isinstance(v, torch.Tensor) for v in c.values()):
                state = c
                break

        if state is None:
            raise RuntimeError("Could not find a valid state_dict inside the checkpoint")

        # Try to infer potential and reward hidden sizes from state keys
        def infer_hidden_sizes(prefix):
            # find keys like f"{prefix}._potential_net.dense0.weight" or reward._reward_net.denseX.weight
            sizes = []
            # collect keys matching dense{n}.weight
            import re
            pattern = re.compile(re.escape(prefix) + r"\.(_potential_net|_reward_net)\.dense(\d+)\.weight")
            found = {}
            for k, v in state.items():
                m = pattern.match(k)
                if m:
                    idx = int(m.group(2))
                    # v is weight tensor with shape (out, in)
                    if isinstance(v, torch.Tensor):
                        found[idx] = int(v.shape[0])
            if not found:
                return ()
            # return sizes in order of idx
            return tuple(found[i] for i in sorted(found.keys()))

        # First try patterns without prefix, then with known names
        potential_sizes = infer_hidden_sizes('potential')
        reward_sizes = infer_hidden_sizes('reward')

        # Fallback: look for explicit potential keys used in some checkpoints
        if not potential_sizes:
            # older keys used 'potential._potential_net' path
            def infer_alt(prefix):
                import re
                pattern = re.compile(re.escape(prefix) + r"\._potential_net\.dense(\d+)\.weight")
                found = {}
                for k, v in state.items():
                    m = pattern.match(k)
                    if m and isinstance(v, torch.Tensor):
                        found[int(m.group(1))] = int(v.shape[0])
                if not found:
                    return ()
                return tuple(found[i] for i in sorted(found.keys()))
            potential_sizes = potential_sizes or infer_alt('') or infer_alt('potential')

        # If still empty, try to detect by looking for 'potential._potential_net.dense0.weight' literal
        if not potential_sizes:
            pat = 'potential._potential_net.dense0.weight'
            if pat in state and isinstance(state[pat], torch.Tensor):
                potential_sizes = (int(state[pat].shape[0]),)

        # If reward sizes empty, try detect reward._reward_net.dense0.weight
        if not reward_sizes:
            pat2 = 'reward._reward_net.dense0.weight'
            if pat2 in state and isinstance(state[pat2], torch.Tensor):
                reward_sizes = (int(state[pat2].shape[0]),)

        # Build net with inferred sizes (or sensible defaults)
        if not potential_sizes:
            # fallback to common sizes used in repository
            potential_sizes = (32, 32)
        if not reward_sizes:
            reward_sizes = (32,)

        from imitation.rewards.reward_nets import BasicShapedRewardNet

        net = BasicShapedRewardNet(
            observation_space=obs_space,
            action_space=act_space,
            reward_hid_sizes=reward_sizes,
            potential_hid_sizes=potential_sizes,
            use_state=True,
            use_action=True,
            use_next_state=False,
            use_done=False,
        )

        try:
            net.load_state_dict(state)
            return net.to(device)
        except RuntimeError as e:
            # Try again with non-strict loading to tolerate extra running-stat keys
            try:
                net.load_state_dict(state, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load state_dict into BasicShapedRewardNet: {e2}")
            print("Warning: loaded state_dict with strict=False; some keys were unexpected/ignored (likely running stat keys)")
            return net.to(device)

    # If it's already a module
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device)

    raise RuntimeError("Unrecognized checkpoint format for reward model")


def rewards_from_net(net, obs, acts, device='cpu'):
    """
    Evaluate net on arrays. obs: (T, obs_dim), acts: (T, act_dim).
    If the net expects next_state and/or done, it will be detected via signature.
    Optionally, obs can be a tuple (obs, next_obs, dones).
    """
    net.eval()
    # allow passing a tuple (obs, next_obs, dones)
    next_obs = None
    dones = None
    if isinstance(obs, tuple) or isinstance(obs, list):
        if len(obs) >= 1:
            obs, next_obs, dones = obs[0], (obs[1] if len(obs) > 1 else None), (obs[2] if len(obs) > 2 else None)
    with torch.no_grad():
        o = torch.tensor(obs, dtype=torch.float32, device=device)
        a = torch.tensor(acts, dtype=torch.float32, device=device)
        # prepare next and done tensors if provided
        next_o = None
        done_t = None
        if next_obs is not None:
            next_o = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if dones is not None:
            done_t = torch.tensor(dones, dtype=torch.float32, device=device)

        # inspect forward signature
        import inspect
        sig = None
        try:
            sig = inspect.signature(net.forward)
        except Exception:
            try:
                sig = inspect.signature(net.__call__)
            except Exception:
                sig = None

        def call_net(o_t, a_t, next_o_t=None, done_t=None):
            # Attempt several common call patterns
            try:
                if next_o_t is not None and done_t is not None:
                    return net(o_t, a_t, next_o_t, done_t)
                if next_o_t is not None:
                    return net(o_t, a_t, next_o_t)
                return net(o_t, a_t)
            except TypeError:
                # fallback: try keyword args
                kwargs = {}
                if 'next_state' in (sig.parameters if sig else {}):
                    kwargs['next_state'] = next_o_t
                if 'done' in (sig.parameters if sig else {}):
                    kwargs['done'] = done_t
                try:
                    return net(o_t, a_t, **kwargs)
                except Exception as e:
                    raise

        try:
            r = call_net(o, a, next_o, done_t)
        except Exception as e:
            # try alternative method names
            if hasattr(net, 'predict_reward'):
                try:
                    if next_o is not None and done_t is not None:
                        r = net.predict_reward(o, a, next_o, done_t)
                    elif next_o is not None:
                        r = net.predict_reward(o, a, next_o)
                    else:
                        r = net.predict_reward(o, a)
                except Exception as e2:
                    raise RuntimeError("Reward net call failed; unknown API: " + str(e2))
            else:
                raise RuntimeError("Reward net call failed; unknown API: " + str(e))

    r = torch.as_tensor(r).view(-1).cpu().numpy()
    return r


def per_trajectory_returns(rewards, dones=None):
    if dones is None:
        return np.array([rewards.sum()])
    returns = []
    start = 0
    for i, d in enumerate(dones):
        if d:
            returns.append(rewards[start:i+1].sum())
            start = i + 1
    if start < len(rewards):
        returns.append(rewards[start:].sum())
    return np.array(returns)


def eval_on_expert(net, expert_npz, device='cpu'):
    data = np.load(expert_npz, allow_pickle=True)
    obs = data['observations']
    acts = data['actions']
    dones = data.get('dones', None)
    L = acts.shape[0]
    # obs usually contains T+1 states; build s_t and s_{t+1}
    if obs.shape[0] >= L + 1:
        obs_t = obs[:L]
        obs_tp1 = obs[1:L+1]
    else:
        obs_t = obs[:L]
        obs_tp1 = None
    r = rewards_from_net(net, (obs_t, obs_tp1, dones), acts, device=device)
    returns = per_trajectory_returns(r, dones)
    return r, returns


def eval_baseline_random_actions(net, obs, acts, action_space, device='cpu'):
    # sample uniform random actions from action_space bounds for each step
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    rng = np.random.RandomState(0)
    acts_rand = rng.uniform(low=low, high=high, size=acts.shape)
    # build next observations if obs contains next states
    next_obs = None
    dones = None
    if isinstance(obs, tuple) or isinstance(obs, list):
        if len(obs) > 1:
            next_obs = obs[1]
        if len(obs) > 2:
            dones = obs[2]
    r = rewards_from_net(net, (obs[0] if isinstance(obs, (list, tuple)) else obs, next_obs, dones), acts_rand, device=device)
    return r


def eval_random_rollouts(net, env, n_rollouts=10, max_steps=500, device='cpu'):
    records = []
    for i in range(n_rollouts):
        obs = env.reset()
        # gym wrapper gives flattened obs array
        obs_buf = []
        next_buf = []
        act_buf = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = env.action_space.sample()
            next_obs, rew, done, info = env.step(action)
            obs_buf.append(obs)
            next_buf.append(next_obs)
            act_buf.append(action)
            obs = next_obs
            steps += 1
        if len(act_buf) == 0:
            continue
        obs_arr = np.asarray(obs_buf, dtype=np.float32)
        act_arr = np.asarray(act_buf, dtype=np.float32)
        next_arr = np.asarray(next_buf, dtype=np.float32)
        r = rewards_from_net(net, (obs_arr, next_arr, None), act_arr, device=device)
        records.append((r, r.sum()))
    return records


def summarize(name, r, returns):
    print(f"--- {name} ---")
    print(f"per-step: mean={r.mean():.4f}, std={r.std():.4f}, sum={r.sum():.4f}")
    if len(returns) > 0:
        print(f"per-trajectory returns: count={len(returns)}, mean={returns.mean():.4f}, std={returns.std():.4f}, min={returns.min():.4f}, max={returns.max():.4f}")
    else:
        print("no returns computed")


def main():
    p = argparse.ArgumentParser(description="Evaluate AIRL reward nets for Franka Lift task")
    p.add_argument('--reward', required=True, help='path to reward checkpoint (state_dict or full model)')
    p.add_argument('--expert-npz', default='trajectories/merged_real_dataset_1.1to1.6_for_training.npz')
    p.add_argument('--device', default='cpu')
    p.add_argument('--baseline', choices=['random_actions','shuffle_actions','random_rollouts','none'], default='random_actions')
    p.add_argument('--n-rollouts', type=int, default=10, help='number of random rollouts when using random_rollouts baseline')
    p.add_argument('--allow-unsafe', action='store_true', help='allow unsafe unpickling for full-model checkpoints')
    args = p.parse_args()

    # prepare env and spaces
    env = make_franka_env()
    obs_space = env.observation_space
    act_space = env.action_space

    # load reward model
    net = load_reward_model(args.reward, obs_space, act_space, device=args.device, allow_unsafe=args.allow_unsafe)

    # eval on expert
    r_exp, returns_exp = eval_on_expert(net, args.expert_npz, device=args.device)
    summarize('Expert', r_exp, returns_exp)

    # baseline: prepare obs_t, obs_tp1, dones tuple so nets expecting next_state get it
    data = np.load(args.expert_npz, allow_pickle=True)
    obs = data['observations']
    acts = data['actions']
    dones = data.get('dones', None)
    L = acts.shape[0]
    if obs.shape[0] >= L + 1:
        obs_t = obs[:L]
        obs_tp1 = obs[1:L+1]
    else:
        obs_t = obs[:L]
        obs_tp1 = None
    obs_for_reward = (obs_t, obs_tp1, dones)

    if args.baseline == 'random_actions':
        r_base = eval_baseline_random_actions(net, obs_for_reward, acts, env.action_space, device=args.device)
        returns_base = per_trajectory_returns(r_base, dones)
        summarize('Baseline - random actions', r_base, returns_base)
    elif args.baseline == 'shuffle_actions':
        acts_shuf = acts.copy()
        rng = np.random.RandomState(1)
        rng.shuffle(acts_shuf)
        r_base = rewards_from_net(net, obs_for_reward, acts_shuf, device=args.device)
        returns_base = per_trajectory_returns(r_base, dones)
        summarize('Baseline - shuffled actions', r_base, returns_base)
    elif args.baseline == 'random_rollouts':
        recs = eval_random_rollouts(net, env, n_rollouts=args.n_rollouts, device=args.device)
        if len(recs) == 0:
            print('No random rollouts collected')
        else:
            sums = np.array([s for (_, s) in recs])
            print(f"Random rollouts: count={len(sums)}, mean_return={sums.mean():.4f}, std={sums.std():.4f}")
    else:
        print('No baseline requested')

    # simple separation metric
    if args.baseline != 'none' and args.baseline != 'random_rollouts':
        try:
            mean_diff = r_exp.mean() - r_base.mean()
            print(f"Expert mean - baseline mean = {mean_diff:.4f}")
            if np.isnan(r_exp).any() or np.isnan(r_base).any():
                print("WARNING: NaNs in rewards")
            if r_exp.mean() <= r_base.mean():
                print("WARNING: expert rewards not higher than baseline; inspect further.")
        except Exception:
            pass

    env.close()


if __name__ == '__main__':
    main()