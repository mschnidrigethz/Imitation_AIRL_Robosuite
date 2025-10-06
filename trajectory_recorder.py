import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Homing, Move, Grasp
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaRobotState
import csv
import os
import time
import threading
import sys
import select
import tty
import termios
import h5py
import numpy as np

"""Real2Sim Trajectory Recorder Node for Franka Emika Panda Robot with Joint and Pose Action Modes.
The node is designed to record expert trajectories on the physical Franka Robot.
They can then be used for Imitation Learning in IsaacSim / IsaacLab."""

class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        
        # Declare parameters for action mode and object position mode
        self.declare_parameter('action_mode', 'joint')  # 'joint' or 'pose'
        self.declare_parameter('object_mode', 'fixed')  # 'fixed' or 'manual'
        
        self.action_mode = self.get_parameter('action_mode').get_parameter_value().string_value
        self.object_mode = self.get_parameter('object_mode').get_parameter_value().string_value
        
        # Validate action mode -> Joint or Position mode
        if self.action_mode not in ['joint', 'pose']:
            self.get_logger().error(f"Invalid action_mode: {self.action_mode}. Must be 'joint' or 'pose'.")
            raise ValueError(f"Invalid action_mode: {self.action_mode}")
        
        # Validate object mode -> Fixed or Manual mode
        if self.object_mode not in ['fixed', 'manual']:
            self.get_logger().error(f"Invalid object_mode: {self.object_mode}. Must be 'fixed' or 'manual'.")
            raise ValueError(f"Invalid object_mode: {self.object_mode}")
        
        self.get_logger().info(f"Action mode set to: {self.action_mode}")
        self.get_logger().info(f"Object position mode set to: {self.object_mode}")
        
        self.recording = False
        self.paused = False
        self.trajectory = []
        self.save_path_hdf5 = os.path.expanduser('~/franka_ros2_ws/src/franka_trajectory_recorder/trajectories/dataset.hdf5')
        self.save_path_csv = os.path.expanduser('~/franka_ros2_ws/src/franka_trajectory_recorder/trajectories/trajectory.csv')
        self.lock = threading.Lock()

        # Sampling rate (20 Hz)
        self.sampling_rate = 20.0  # Hz
        self.sampling_period = 1.0 / self.sampling_rate  # Seconds

        # Initialize the robot root pose and velocity
        self.initial_state = {
            'joint_position': None,
            'joint_velocity': None,
            'root_pose': None,
            'root_velocity': None
        }

        # Initialize placeholders for latest joint positions and velocities
        self.latest_joint_positions = None
        self.latest_joint_velocities = None

        # Initialize rigid objects with default positions
        self.initialize_rigid_objects()

        # Subscriptions to joint states and gripper states

        # Subsribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10)
        
        # Subscribe to gripper state
        self.gripper_subscription = self.create_subscription(
            JointState, 
            '/fr3_gripper/joint_states', 
            self.gripper_state_callback, 
            10)
        
        # Subscribe to Franka robot state (ee pose and velocity)
        self.franka_state = self.create_subscription(
            FrankaRobotState, 
            '/franka_robot_state_broadcaster/robot_state', 
            self.franka_state_callback, 
            10)

        # Timer for sampling at 20 Hz -> Call self.sample_trajectory every 0.05 seconds
        self.timer = self.create_timer(self.sampling_period, self.sample_trajectory)

        # Gripper state
        self.gripper_state = 'open'
        self.gripper_goal_state = 'open'

        # Gripper control parameters
        self.gripper_max_width = 0.08
        self.gripper_speed = 0.5
        self.gripper_force = 50.0
        self.gripper_epsilon_inner = 0.05
        self.gripper_epsilon_outer = 0.05

        # Action clients for gripper control
        self.homing_client = ActionClient(self, Homing, '/fr3_gripper/homing')
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move')
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')

        # Wait for action servers
        self.wait_for_action_server(self.homing_client, 'Homing')
        self.wait_for_action_server(self.move_client, 'Move')
        self.wait_for_action_server(self.grasp_client, 'Grasp')

        # Perform initial homing
        self.home_gripper()

        # Display instructions
        self.display_instructions()

        # Start keyboard listener
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def initialize_rigid_objects(self):
        """Initialize rigid objects with default hardcoded positions."""
        self.default_rigid_objects = {
            # blue cube
            'object': {
                'root_pose': np.array([[0.4, 0.2, 0.0203, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                'root_velocity': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            }
        }
        
        
        # Start with default positions
        self.rigid_objects = self.default_rigid_objects.copy()
        #self.rigid_objects = {}
        #for name, data in self.default_rigid_objects.items():
        #    self.rigid_objects[name] = {
        #        'root_pose': data['root_pose'].copy(),      # ← WICHTIG: .copy() für numpy Arrays
        #        'root_velocity': data['root_velocity'].copy()  # ← WICHTIG: .copy() für numpy Arrays
        #    }
            
    def display_instructions(self):
        """Display user instructions based on the current mode."""
        mode_info = f"Action: {self.action_mode.upper()}, Objects: {self.object_mode.upper()}"
        action_format = "Joint positions + gripper" if self.action_mode == 'joint' else "End-effector pose + gripper"
        
        instructions = f"""
================ Trajectory Recorder ({mode_info}) =================
Press the following keys for corresponding actions:
  [r] - Start/Pause recording
  [f] - Finish recording and save trajectory
  [b] - Open/Close gripper state"""
        
        if self.object_mode == 'manual':
            instructions += "\n  [o] - Set custom object positions before recording"
        
        instructions += f"""
Action format: {action_format}
Current object positions:"""
        
        for name, data in self.rigid_objects.items():
            pos = data['root_pose'][0, :3]
            instructions += f"\n  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        
        instructions += "\n====================================================="
        
        self.get_logger().info(instructions)

    def get_float_input(self, prompt, default_value=None):
        """Get float input from user with validation and default value support."""
        while True:
            try:
                if default_value is not None:
                    user_input = input(f"{prompt} (default: {default_value:.3f}): ").strip()
                    if user_input == "":
                        return default_value
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                return float(user_input)
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return None

    def set_custom_object_positions(self):
        """Interactive method to set custom object positions."""
        print("\n" + "="*60)
        print("CUSTOM OBJECT POSITION SETUP")
        print("="*60)
        print("Enter new positions for each cube.")
        print("Press Enter to keep current position, or type 'skip' to skip this cube.")
        print("Press Ctrl+C at any time to cancel and keep current positions.")
        print("-"*60)
        
        try:
            new_positions = {}
            
            cube_names = {
                'object': 'Blue Cube'
            }
            
            for cube_id, cube_name in cube_names.items():
                print(f"\n{cube_name} ({cube_id}):")
                current_pos = self.rigid_objects[cube_id]['root_pose'][0, :3]
                print(f"Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                
                # Ask if user wants to change this cube's position
                change = input("Change position? (y/n/skip): ").strip().lower()
                if change in ['n', 'no', 'skip']:
                    new_positions[cube_id] = current_pos.copy()
                    continue
                elif change not in ['y', 'yes', '']:
                    print("Invalid input. Skipping this cube.")
                    new_positions[cube_id] = current_pos.copy()
                    continue
                
                # Get new coordinates
                print("Enter new coordinates:")
                x = self.get_float_input("  X", current_pos[0])
                if x is None:
                    return  # User cancelled
                
                y = self.get_float_input("  Y", current_pos[1])
                if y is None:
                    return  # User cancelled
                
                z = self.get_float_input("  Z", current_pos[2])
                if z is None:
                    return  # User cancelled
                
                new_positions[cube_id] = np.array([x, y, z], dtype=np.float32)
                print(f"  → New position: [{x:.3f}, {y:.3f}, {z:.3f}]")
            
            # Apply new positions
            for cube_id, new_pos in new_positions.items():
                self.rigid_objects[cube_id]['root_pose'][0, :3] = new_pos
            
            print("\n" + "="*60)
            print("OBJECT POSITIONS UPDATED SUCCESSFULLY!")
            print("="*60)
            print("New object positions:")
            for cube_id, cube_name in cube_names.items():
                pos = self.rigid_objects[cube_id]['root_pose'][0, :3]
                print(f"  {cube_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled. Object positions unchanged.")
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Object positions unchanged.")

    def reset_to_default_positions(self):
        """Reset object positions to default hardcoded values."""
        self.rigid_objects = self.default_rigid_objects.copy()
        self.get_logger().info("Object positions reset to default values.")

        # DEEP COPY: Erstelle komplett neue numpy Arrays
        #self.rigid_objects = {}
        #for name, data in self.default_rigid_objects.items():
        #    self.rigid_objects[name] = {
        #        'root_pose': data['root_pose'].copy(),      # ← WICHTIG: .copy() für numpy Arrays
        #        'root_velocity': data['root_velocity'].copy()  # ← WICHTIG: .copy() für numpy Arrays
        #    }
        self.get_logger().info("Object positions reset to default values.")
        
        
    # ----------------------------- Callbacks for subscriptions -----------------------------

    def joint_state_callback(self, msg):
        # Store the latest joint positions and velocities
        self.latest_joint_positions = list(msg.position)[:7]  # First 7 joint positions
        self.latest_joint_velocities = list(msg.velocity)[:7]  # First 7 joint velocities

    def gripper_state_callback(self, msg):
        self.gripper_state = list(msg.position)  # Store the gripper state dynamically
        self.get_logger().debug(f"Gripper state updated: {self.gripper_state}")

    def franka_state_callback(self, msg: FrankaRobotState):
        # Get the ee-position and orientation from the FrankaRobotState message
        position = msg.o_t_ee.pose.position
        orientation = msg.o_t_ee.pose.orientation
        # Set robot_root_pose
        self.robot_root_pose = np.array([
            position.x, position.y, position.z,
            orientation.x, orientation.y, orientation.z, orientation.w
        ], dtype=np.float32)
        # TODO: Do we still have to populate this array?
        self.robot_root_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.get_logger().debug(f"Updated root_pose: {self.robot_root_pose}, root_velocity: {self.robot_root_velocity}")

    def sample_trajectory(self):
        """Record trajectory data at a fixed rate (20 Hz)."""
        if self.recording and not self.paused:
            with self.lock:
                timestamp = time.time()

                # Determine the gripper action as a binary value
                gripper_action = 1.0 if self.gripper_goal_state == 'open' else 0.0

                # Append the data to the trajectory
                self.trajectory.append({
                    'timestamp': timestamp,
                    'joint_positions': self.latest_joint_positions,
                    'joint_velocities': self.latest_joint_velocities,
                    'gripper_action': gripper_action,
                    'robot_root_pose': self.robot_root_pose,
                })
            self.get_logger().debug(f"Sampled trajectory at {timestamp}: {self.latest_joint_positions}, {self.latest_joint_velocities}")

    # Action generation based on the selected action mode: 'joint' or 'pose'
    def generate_actions(self):
        """Generate actions based on the selected action mode."""
        actions = []
        
        if self.action_mode == 'joint':
            # Joint position mode: [joint1, joint2, ..., joint7, gripper_command]
            for entry in self.trajectory:
                absolute_joint_positions = entry['joint_positions']
                gripper_command = 1 if entry['gripper_action'] == 1.0 else -1
                absolute_action = absolute_joint_positions + [gripper_command]
                actions.append(absolute_action)
                
            # Set the first action to be the initial joint position (starting state)
            if actions and self.initial_state['joint_position'] is not None:
                initial_joint_positions = self.initial_state['joint_position'][:7]  # First 7 joints only
                actions[0] = initial_joint_positions + [1]  # 1 for gripper open (starting position)
                
        elif self.action_mode == 'pose':
            # End-effector pose mode: [x, y, z, qw, qx, qy, qz, gripper_command]
            for entry in self.trajectory:
                pose = entry['robot_root_pose']
                # Reorder quaternion from [x, y, z, qx, qy, qz, qw] to [x, y, z, qw, qx, qy, qz]
                pose_action = [pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]]
                gripper_command = 1 if entry['gripper_action'] == 1.0 else -1
                absolute_action = pose_action + [gripper_command]
                actions.append(absolute_action)
                
            # Set the first action to be the initial pose (starting state)
            if actions and self.initial_state['root_pose'] is not None:
                initial_pose = self.initial_state['root_pose']
                # Reorder quaternion for initial pose as well
                initial_pose_action = [initial_pose[0], initial_pose[1], initial_pose[2], 
                                     initial_pose[6], initial_pose[3], initial_pose[4], initial_pose[5]]
                actions[0] = initial_pose_action + [1]  # 1 for gripper open (starting position)
        
        return actions

    def calculate_relative_quaternion(self, q1, q2):
        """
        Calculate the relative quaternion between two quaternions q1 and q2.
        """
        # Convert q1 to a numpy array
        q1 = np.array(q1)
        q2 = np.array(q2)

        # Conjugate of q1
        q1_conjugate = np.array([q1[0], -q1[1], -q1[2], -q1[3]])

        # Quaternion multiplication: q_relative = q2 * q1_conjugate
        q_relative = np.array([
            q2[0] * q1_conjugate[0] - q2[1] * q1_conjugate[1] - q2[2] * q1_conjugate[2] - q2[3] * q1_conjugate[3],
            q2[0] * q1_conjugate[1] + q2[1] * q1_conjugate[0] + q2[2] * q1_conjugate[3] - q2[3] * q1_conjugate[2],
            q2[0] * q1_conjugate[2] - q2[1] * q1_conjugate[3] + q2[2] * q1_conjugate[0] + q2[3] * q1_conjugate[1],
            q2[0] * q1_conjugate[3] + q2[1] * q1_conjugate[2] - q2[2] * q1_conjugate[1] + q2[3] * q1_conjugate[0]
        ])

        return q_relative.tolist()

    # Keyboard listener for user inputs
    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'r':
                        self.toggle_recording()
                    elif key == 'f':
                        self.finish_recording()
                    elif key == 'b':
                        if self.gripper_goal_state == 'open':
                            self.close_gripper()
                            self.gripper_goal_state = 'closed'
                        elif self.gripper_goal_state == 'closed':
                            self.open_gripper()
                            self.gripper_goal_state = 'open'
                    elif key == 'o' and self.object_mode == 'manual':
                        # Temporarily restore terminal settings for input
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        
                        #zz------------------- RESET ZU DEFAULT-WERTEN BEVOR CUSTOM POSITIONS
                        # self.reset_to_default_positions()  # ← Das ist der Schlüssel!
                        
                        self.set_custom_object_positions()
                        self.display_instructions()
                        # Restore cbreak mode
                        tty.setcbreak(sys.stdin.fileno())
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.paused = False
            self.get_logger().info("Recording started.")

            # Capture initial state of the robot when recording starts
            with self.lock:
                if self.latest_joint_positions is not None and self.latest_joint_velocities is not None:
                    self.initial_state['joint_position'] = self.latest_joint_positions + [self.gripper_state[0], self.gripper_state[0]]
                    self.initial_state['joint_velocity'] = np.zeros(9, dtype=np.float32)
                    self.initial_state['root_pose'] = self.robot_root_pose
                    self.initial_state['root_velocity'] = np.zeros(6, dtype=np.float32)
                else:
                    self.get_logger().warning("No joint states available to initialize the recording.")
        elif self.recording and not self.paused:
            self.paused = True
            self.get_logger().info("Recording paused.")
        elif self.recording and self.paused:
            self.paused = False
            self.get_logger().info("Recording resumed.")

    def finish_recording(self):
        if self.recording:
            self.recording = False
            self.paused = False
            self.get_logger().info("Recording finished. Saving trajectory...")
            self.save_trajectory_csv()
            self.save_trajectory()
        else:
            self.get_logger().info("No recording in progress to finish.")

    def save_trajectory_csv(self):
        """Save trajectory data to CSV format supporting both joint and pose modes."""
        with self.lock:
            if not self.trajectory:
                self.get_logger().warning("No trajectory data to save to CSV.")
                return

            try:
                with open(self.save_path_csv, 'w', newline='') as csvfile:
                    if self.action_mode == 'joint':
                        # Joint mode CSV format
                        fieldnames = [
                            'timestamp', 'joint_positions', 'joint_velocities', 
                            'gripper_action', 'gripper_goal_state'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for entry in self.trajectory:
                            writer.writerow({
                                'timestamp': entry['timestamp'],
                                'joint_positions': ','.join(map(str, entry['joint_positions'])),
                                'joint_velocities': ','.join(map(str, entry['joint_velocities'])),
                                'gripper_action': entry['gripper_action'],
                                'gripper_goal_state': 'open' if entry['gripper_action'] == 1.0 else 'closed'
                            })
                    
                    elif self.action_mode == 'pose':
                        # Pose mode CSV format
                        fieldnames = [
                            'timestamp', 'joint_positions', 'joint_velocities',
                            'eef_position', 'eef_orientation', 'gripper_action', 'gripper_goal_state'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for entry in self.trajectory:
                            pose = entry['robot_root_pose']
                            eef_pos = pose[:3]  # x, y, z
                            eef_orient = pose[3:]  # qx, qy, qz, qw
                            
                            writer.writerow({
                                'timestamp': entry['timestamp'],
                                'joint_positions': ','.join(map(str, entry['joint_positions'])),
                                'joint_velocities': ','.join(map(str, entry['joint_velocities'])),
                                'eef_position': ','.join(map(str, eef_pos)),
                                'eef_orientation': ','.join(map(str, eef_orient)),
                                'gripper_action': entry['gripper_action'],
                                'gripper_goal_state': 'open' if entry['gripper_action'] == 1.0 else 'closed'
                            })

                self.get_logger().info(f"Trajectory saved to CSV: {self.save_path_csv} in {self.action_mode} mode")
                
            except Exception as e:
                self.get_logger().error(f"Failed to save CSV trajectory: {e}")

    # ----------------------------- Save trajectory to HDF5 -----------------------------
    # The data is stored in the following structure:
    # /data
    #   ├── demo_0
    #   │   ├── num_samples
    #   │   ├── success
    #   │   ├── actions
    #   │   ├── initial_state
    #   │   │   ├── articulation
    #   │   │   │   ├── robot
    #   │   │   │       ├── joint_position
    #   │   │   │       ├── joint_velocity
    #   │   │   │       ├── root_pose
    #   │   │   │       ├── root_velocity
    #   │   │       ├── rigid_object
    #   │   │   │       ├── object
    #   │   ├── obs
    #   │   │   ├── actions
    #   │   │   ├── cube_orientations
    #   │   │   ├── cube_positions
    #   │   │   ├── eef_pos
    #   │   │   ├── eef_quat
    #   │   │   ├── gripper_pos
    #   │   │   ├── joint_pos
    #   │   │   ├── joint_vel
    #   │   │   ├── object
    def save_trajectory(self):
        with self.lock:
            with h5py.File(self.save_path_hdf5, 'a') as hdf5file:
                # Ensure the /data group exists
                if 'data' not in hdf5file:
                    data_group = hdf5file.create_group('data')
                    # Set environment arguments for the dataset: Isaac-Stack-Cube-Franka-IK-Abs-v0 for pose mode & Isaac-Stack-Cube-Franka-v0 for joint mode
                    env_name = "Isaac-Stack-Cube-Franka-IK-Abs-v0" if self.action_mode == 'pose' else "Isaac-Stack-Cube-Franka-v0"
                    data_group.attrs['env_args'] = f'{{"env_name": "{env_name}", "type": 2}}'
                    data_group.attrs['total'] = 0
                else:
                    data_group = hdf5file['data']

                # Determine the next demo group name
                demo_index = len(data_group.keys())
                demo_group_name = f'demo_{demo_index}'
                demo_group = data_group.create_group(demo_group_name)

                # Add attributes to the demo group
                demo_group.attrs['num_samples'] = len(self.trajectory)
                demo_group.attrs['success'] = np.bool_(True)

                # Generate actions based on the selected mode
                actions = self.generate_actions()
                demo_group.create_dataset('actions', data=np.array(actions, dtype=np.float32))

                # Log action format and object positions for debugging
                if actions:
                    action_format = "Joint positions" if self.action_mode == 'joint' else "End-effector pose"
                    self.get_logger().info(f"Saved actions in {action_format} format. Action shape: {np.array(actions).shape}")
                    
                    # Log object positions used in this demo
                    self.get_logger().info("Object positions used in this demo:")
                    for name, data in self.rigid_objects.items():
                        pos = data['root_pose'][0, :3]
                        self.get_logger().info(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

                # -------------------------- Add the initial_state group --------------------------
                initial_state = demo_group.create_group('initial_state')
                articulation = initial_state.create_group('articulation')
                robot = articulation.create_group('robot')

                # Save joint_position with shape (1, 9)
                robot.create_dataset(
                    'joint_position',
                    data=np.expand_dims(np.array(self.initial_state['joint_position'], dtype=np.float32), axis=0)
                )

                # Save joint_velocity with shape (1, 9)
                robot.create_dataset(
                    'joint_velocity',
                    data=np.expand_dims(np.array(self.initial_state['joint_velocity'], dtype=np.float32), axis=0)
                )

                # Save root_pose with shape (1, 7)
                robot.create_dataset(
                    'root_pose',
                    # Default robot root pose is [0.0, 0.0, 0.0, 1, 0, 0, 0]
                    data = np.expand_dims(np.array([0.0, 0.0, 0.0, 1, 0, 0, 0], dtype=np.float32), axis=0)   
                )

                # Save root_velocity with shape (1, 6)
                robot.create_dataset(
                    'root_velocity',
                    # Default robot root velocity is [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    data = np.expand_dims(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), axis=0)
                )

                # -------------------------- Add the rigid_object group --------------------------
                rigid_object = initial_state.create_group('rigid_object')
                for name, data in self.rigid_objects.items():
                    cube = rigid_object.create_group(name)
                    cube.create_dataset('root_pose', data=data['root_pose'])
                    cube.create_dataset('root_velocity', data=data['root_velocity'])

                # --------------------------- Add the obs group ----------------------------------             
                obs = demo_group.create_group('obs')
                num_samples = len(self.trajectory)

                # Dynamically populate the 'actions' dataset (same as above)
                obs.create_dataset('actions', data=np.array(actions, dtype=np.float32))

                # Dynamically populate the 'cube_orientations' and 'cube_positions' datasets
                cube_orientations = []
                cube_positions = []
                for _ in range(num_samples):
                    # Extract orientations (quaternion: qx, qy, qz, qw) for object only
                    orientation_object = self.rigid_objects['object']['root_pose'][0, 3:]
                    cube_orientations.append(orientation_object)

                    # Extract positions (x, y, z) for object only
                    position_object = self.rigid_objects['object']['root_pose'][0, :3]
                    cube_positions.append(position_object)

                obs.create_dataset('cube_orientations', data=np.array(cube_orientations, dtype=np.float32))
                obs.create_dataset('cube_positions', data=np.array(cube_positions, dtype=np.float32))

                # Dynamically populate the 'eef_pos', 'eef_quat' datasets
                # Extract x,y,z
                eef_pos = [entry['robot_root_pose'][:3] for entry in self.trajectory]
                # Extract qx,qy,qz,qw
                eef_quat = [entry['robot_root_pose'][3:] for entry in self.trajectory]
                # Create datasets
                obs.create_dataset('eef_pos', data=np.array(eef_pos), dtype=np.float32)
                obs.create_dataset('eef_quat', data=np.array(eef_quat), dtype=np.float32)

                # Dynamically populate the 'gripper_pos' dataset using absolute positions
                gripper_pos = [entry['joint_positions'] for entry in self.trajectory]  # Use full 7 joint positions
                obs.create_dataset('gripper_pos', data=np.array(gripper_pos, dtype=np.float32))
                
                # Dynamically populate the 'joint_pos', 'joint_vel' datasets
                joint_pos = [entry['joint_positions'] for entry in self.trajectory]
                joint_vel = [entry['joint_velocities'] for entry in self.trajectory]
                obs.create_dataset('joint_pos', data=np.array(joint_pos, dtype=np.float32))
                obs.create_dataset('joint_vel', data=np.array(joint_vel, dtype=np.float32))

                # ------------------------ Add the 'object' dataset --------------------------
                # The object dataset in the obs group contains info about the stae of objects in the env at each timestep
                # Dataset is storing information for three cubes each with 13 features
                # Obtain the data from the vision system
                # The features are: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
                obs.create_dataset('object', data=np.zeros((num_samples, 39), dtype=np.float32))

                # --------------------------- Add the states group ----------------------------------
                states = demo_group.create_group('states')

                # Articulation group
                articulation_states = states.create_group('articulation')
                robot_states = articulation_states.create_group('robot')

                # Populate robot states with absolute positions
                joint_positions = [entry['joint_positions'] for entry in self.trajectory]
                joint_velocities = [entry['joint_velocities'] for entry in self.trajectory]
                # Fill with same constant base pose [0.5, 0.0, 0.0, 1, 0, 0, 0 ]
                root_poses = np.tile([0.5, 0.0, 0.0, 1, 0, 0, 0], (len(self.trajectory), 1))
                root_velocities = [self.robot_root_velocity for _ in self.trajectory]  # Assuming constant velocity for now

                robot_states.create_dataset('joint_position', data=np.array(joint_positions, dtype=np.float32))
                robot_states.create_dataset('joint_velocity', data=np.array(joint_velocities, dtype=np.float32))
                robot_states.create_dataset('root_pose', data=np.array(root_poses, dtype=np.float32))
                robot_states.create_dataset('root_velocity', data=np.array(root_velocities, dtype=np.float32))

                # Rigid object group
                rigid_object_states = states.create_group('rigid_object')
                for name, data in self.rigid_objects.items():
                    cube_states = rigid_object_states.create_group(name)
                    cube_root_poses = np.tile(data['root_pose'], (len(self.trajectory), 1))
                    cube_root_velocities = np.tile(data['root_velocity'], (len(self.trajectory), 1))
                    cube_states.create_dataset('root_pose', data=cube_root_poses)
                    cube_states.create_dataset('root_velocity', data=cube_root_velocities)

                # Update the total number of samples in the data group
                if 'total' in data_group.attrs:
                    data_group.attrs['total'] += len(self.trajectory)
                else:
                    data_group.attrs['total'] = len(self.trajectory)

            self.get_logger().info(f"Trajectory saved to {self.save_path_hdf5} under group {demo_group_name} in {self.action_mode} mode")
            self.trajectory = []

    def wait_for_action_server(self, client, name):
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
            self.get_logger().info(f'{name} action server not available, waiting again...')
        if rclpy.ok():
            self.get_logger().info(f'{name} action server found.')
        else:
            self.get_logger().error(f'ROS shutdown while waiting for {name} server.')
            raise SystemExit('ROS shutdown')

    # ----------------------------- Gripper control methods -----------------------------
    def home_gripper(self):
        self.get_logger().info("Sending homing goal...")
        goal_msg = Homing.Goal()
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_state = 'open'
        self.get_logger().info("Homing goal sent.")

    def close_gripper(self):
        self.get_logger().info("Sending close gripper goal...")
        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0
        goal_msg.speed = self.gripper_speed
        goal_msg.force = self.gripper_force
        goal_msg.epsilon.inner = self.gripper_epsilon_inner
        goal_msg.epsilon.outer = self.gripper_epsilon_outer
        self.grasp_client.send_goal_async(goal_msg)
        self.gripper_state = 'closed'
        self.get_logger().info("Gripper closed.")

    def open_gripper(self):
        self.get_logger().info("Sending open gripper goal...")
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        self.move_client.send_goal_async(goal_msg)
        self.gripper_state = 'open'
        self.get_logger().info("Gripper opened.")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Trajectory Recorder.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()