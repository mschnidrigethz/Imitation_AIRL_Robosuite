import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True
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

print("Observation space shape:", env.observation_spec())
print("Observation keys:", env.obs_keys)