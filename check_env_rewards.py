from robosuite.wrappers.gym_wrapper import GymWrapper
import robosuite as suite
import numpy as np

env = suite.make(env_name="Lift", robots="Panda", has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True)
env = GymWrapper(env)
obs = env.reset()
total = 0.0
for _ in range(200):
    a = env.action_space.sample()
    out = env.step(a)
    # Support both Gym (obs, reward, done, info) and Gymnasium (obs, reward, terminated, truncated, info)
    if len(out) == 4:
        obs, r, done, info = out
    elif len(out) == 5:
        obs, r, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise RuntimeError(f"Unexpected env.step return signature with length {len(out)}")
    total += r
    if done:
        break
print("Random policy env total reward:", total)