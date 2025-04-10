import motornet as mn
import numpy as np
import torch as th
import matplotlib as plt
import math

class Arm:

    def __init__(self, arm_type, integrator='rk4', verbose=False):

        self.integrator = 'rk4'

        if arm_type == 'arm210':
            self.effector = self.build_arm210(verbose)
        elif arm_type == 'arm26':
            self.effector = self.build_arm_26(verbose)

    def build_arm210(self, verbose):

        arm210 = mn.effector.Effector(
        skeleton=mn.skeleton.TwoDofArm(),
        muscle=mn.muscle.RigidTendonHillMuscleThelen(),
        integration_method=self.integrator
        )

        arm210.add_muscle(
        path_fixation_body=[0., 1.],
        path_coordinates=[[-.15, .03], [.094, 0.017]],
        name='pectoralis',
        max_isometric_force=838,
        tendon_length=.039,
        optimal_muscle_length=.134,
        normalized_slack_muscle_length=1.48)

        arm210.add_muscle(
        path_fixation_body=[0., 1.],
        path_coordinates=[[-.034, .022], [.144, 0.01]],
        name='clavicular deltoid',
        max_isometric_force=680,
        tendon_length=.039,
        optimal_muscle_length=.104,
        normalized_slack_muscle_length=1.4)

        arm210.add_muscle(
        path_fixation_body=[0., 0., 1.],
        path_coordinates=[[.14, 0.], [.05, -.00], [0.153, 0.]],
        name='deltoid',
        max_isometric_force=1207,
        tendon_length=.066,
        optimal_muscle_length=.140,
        normalized_slack_muscle_length=1.52)

        arm210.add_muscle(
        path_fixation_body=[0., 0., 1.],
        path_coordinates=[[.1, 0.], [.05, -.03], [0.062, 0.004]],
        name='teres major',
        max_isometric_force=1207,
        tendon_length=.066,
        optimal_muscle_length=.068,
        normalized_slack_muscle_length=1.65)

        arm210.add_muscle(
        path_fixation_body=[1., 2.],
        path_coordinates=[[0.23, 0.001], [0.231, 0.01]],
        name='brachioradialis',
        max_isometric_force=1422,
        tendon_length=.172,
        optimal_muscle_length=.092,
        normalized_slack_muscle_length=1.43)

        arm210.add_muscle(
        path_fixation_body=[1., 1., 2.],
        path_coordinates=[[0.03, 0.], [0.138, -0.019], [-0.04, -0.017]],
        name='tricepslat',
        max_isometric_force=1549,
        tendon_length=.187,
        optimal_muscle_length=.093,
        normalized_slack_muscle_length=1.45)

        arm210.add_muscle(
        path_fixation_body=[0., 2.],
        path_coordinates=[[-0.052, 0.033], [0.044, 0.001]],
        name='biceps',
        max_isometric_force=414,
        tendon_length=.204,
        optimal_muscle_length=.137,
        normalized_slack_muscle_length=1.5)

        arm210.add_muscle(
        path_fixation_body=[0., 2.],
        path_coordinates=[[0.02, -0.028], [-0.04, -0.017]],
        name='tricepslong',
        max_isometric_force=603,
        tendon_length=0.217,
        optimal_muscle_length=0.127,
        normalized_slack_muscle_length=1.4)

        arm210.add_muscle(
        path_fixation_body=[1., 2.],
        path_coordinates=[[0.306, -0.011], [0.003, -0.025]],
        name='anconeus',
        max_isometric_force=300,
        tendon_length=0.01,
        optimal_muscle_length=0.015,
        normalized_slack_muscle_length=1.72)

        arm210.add_muscle(
        path_fixation_body=[1., 2.],
        path_coordinates=[[0.277, 0.], [0.075, 0.02]],
        name='prot',
        max_isometric_force=700,
        tendon_length=0.02,
        optimal_muscle_length=0.058,
        normalized_slack_muscle_length=1.48)

        if verbose:
            arm210.print_muscle_wrappings()

        return arm210

    def build_arm26(self, verbose):

        arm26 = mn.effector.RigidTendonArm26(
        muscle=mn.muscle.MujocoHillMuscle(),
        integration_method=self.integrator)

        if verbose:
            arm26.print_muscle_wrappings()

        return arm26

    def print_effector_state(self, batch_size):
        self.effector.reset(options={"batch_size": batch_size})

        for key, state in self.effector.states.items():
            print(key + " shape: " + " " * (10-len(key)), state.shape)
        
        print()

    def print_muscle_state(self):

        features = self.effector.muscle.state_name
        for n, feature in enumerate(features):
            print("feature " + str(n) + ": ", feature)

    def visualize_muscle_len(self):

        n_states = 21  # grid resolution
        sho, elb = np.meshgrid(
            np.linspace(self.effector.pos_lower_bound[0], self.effector.pos_upper_bound[0], n_states).astype('float32'),
            np.linspace(self.effector.pos_lower_bound[1], self.effector.pos_upper_bound[1], n_states).astype('float32'))
        sho, elb = th.tensor(sho), th.tensor(elb)
        self.effector.reset(options={"joint_state": th.stack([sho.reshape(-1), elb.reshape(-1)], axis=1)})
        mstate = self.effector.states["muscle"].numpy().reshape((n_states, n_states, -1, self.effector.n_muscles))
        gstate = self.effector.states["geometry"].numpy().reshape((n_states, n_states, -1, self.effector.n_muscles))
        sho, elb = th.tensor(sho), th.tensor(elb)

        fig = plt.figure(figsize=(22, 7))
        fig.patch.set_facecolor('xkcd:white')
        # fig.set_tight_layout(True)
        for m in range(self.effector.n_muscles):
            ax = fig.add_subplot(2, 5, m+1, projection='3d')
            ax.plot_surface(sho, elb, mstate[:, :, 1, m], cmap=plt.get_cmap('coolwarm'))
            ax.set_xlabel('shoulder angle (rad)')
            ax.set_ylabel('elbow angle (rad)')
            ax.set_zlabel('muscle length (m)')
            ax.view_init(18, -40)
            ax.locator_params(nbins=6)
            plt.title(self.effector.muscle_name[m])
        plt.show()

    def visualize_muscle_moments(self):
        # create a grid of joint angles
        n_states = 21  # grid resolution
        sho, elb = np.meshgrid(
            np.linspace(self.effector.pos_lower_bound[0], self.effector.pos_upper_bound[0], n_states).astype('float32'),
            np.linspace(self.effector.pos_lower_bound[1], self.effector.pos_upper_bound[1], n_states).astype('float32'))
        sho, elb = th.tensor(sho), th.tensor(elb)

        self.effector.reset(options={"joint_state": th.stack([sho.reshape(-1), elb.reshape(-1)], axis=1)})
        mstate = self.effector.states["muscle"].numpy().reshape((n_states, n_states, -1, self.effector.n_muscles))
        gstate = self.effector.states["geometry"].numpy().reshape((n_states, n_states, -1, self.effector.n_muscles))
        sho, elb = th.tensor(sho), th.tensor(elb)

        fig = plt.figure(figsize=(22, 7))
        fig.patch.set_facecolor('xkcd:white')
        # fig.set_tight_layout(True)
        for m in range(self.effector.n_muscles):
            ax = fig.add_subplot(2, 5, m+1, projection='3d')
            ax.plot_surface(sho, elb, gstate[:, :, 2, m] * 100, cmap=plt.get_cmap('coolwarm'))
            ax.plot_surface(sho, elb, gstate[:, :, 3, m] * 100, cmap=plt.get_cmap('coolwarm'))
            ax.set_xlabel('shoulder angle (rad)')
            ax.set_ylabel('elbow angle (rad)')
            ax.set_zlabel('moment arm (cm)')
            ax.view_init(18, -40)
            ax.locator_params(nbins=6)
            plt.title(self.effector.muscle_name[m])
        plt.show()

class CustomReachEnv(mn.environment.Environment):
    def __init__(self, effector, **kwargs):
        # Set max episode duration to 5 seconds
        super().__init__(effector, max_ep_duration=5.0, **kwargs)
        self.device = effector.device
        # Task parameters
        self.goal_radius = 0.01  # 1cm in meters
        self.cue_delay = 0.2     # 200ms before movement allowed
        self.hold_threshold = int(0.8 / self.dt)  # 800ms in timesteps
        
        # State trackers
        self.current_time = 0.0
        self.hold_counter = None
        self.goal_position = None

    def reset(self, seed=None, options=None):
        options = options or {}
        batch_size = options.get("batch_size", 1)
        
        # Determine the expected number of joints from the effector's skeleton.
        n_joints_expected = self.effector.skeleton.state_dim  # e.g. 4

        if hasattr(self.effector, "q_init") and (self.effector.q_init is not None):
            if self.effector.q_init.numel() == n_joints_expected:
                q_target = self.effector.q_init
                if q_target.dim() == 1:
                    q_target = q_target.unsqueeze(0).expand(batch_size, -1)
            else:
                lb = self.effector.pos_lower_bound
                ub = self.effector.pos_upper_bound
                # Convert to torch tensors if needed.
                if isinstance(lb, np.ndarray):
                    lb = th.tensor(lb, device=self.effector.device, dtype=th.float32)
                if isinstance(ub, np.ndarray):
                    ub = th.tensor(ub, device=self.effector.device, dtype=th.float32)
                q_target = th.rand(batch_size, lb.numel(), device=self.effector.device) * (ub - lb) + lb
        else:
            lb = self.effector.pos_lower_bound
            ub = self.effector.pos_upper_bound
            if isinstance(lb, np.ndarray):
                lb = th.tensor(lb, device=self.effector.device, dtype=th.float32)
            if isinstance(ub, np.ndarray):
                ub = th.tensor(ub, device=self.effector.device, dtype=th.float32)
            q_target = th.rand(batch_size, lb.numel(), device=self.effector.device) * (ub - lb) + lb

        cart_target = self.effector.joint2cartesian(q_target)
        q_start = q_target + th.randn_like(q_target) * 0.01

        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)

        self.goal_position = cart_target
        self.current_time = 0.0
        self.hold_counter = th.zeros(batch_size, device=self.effector.device)
        info["goal"] = cart_target
        return obs, info



    def step(self, action, deterministic=False):
        # 1. Handle pre-cue phase (first 200ms)
        if self.current_time < self.cue_delay:
            action = th.zeros_like(action)  # Freeze actions before cue

        # 2. Perform environment step
        obs, reward, terminated, truncated, info = super().step(
            action, deterministic=deterministic
        )
        
        # 3. Update timers
        self.current_time += self.dt
        
        # 4. Calculate target proximity
        fingertip_pos = info["states"]["fingertip"]
        dist_to_target = th.norm(fingertip_pos - self.goal_position, dim=-1)
        in_target = dist_to_target < self.goal_radius
        
        # 5. Update hold counter
        self.hold_counter = th.where(in_target, self.hold_counter + 1, 0)
        
        # 6. Early termination check
        terminated = terminated | (self.hold_counter >= self.hold_threshold)
        
        return obs, reward, terminated, truncated, info

    def get_obs(self, action=None, deterministic=False):
        # Add go-cue signal to observations
        base_obs = super().get_obs(action, deterministic)
        cue_signal = (self.current_time >= self.cue_delay).float().unsqueeze(-1)
        return th.cat([base_obs, cue_signal], dim=-1)

class CenterOutReachEnv(mn.environment.Environment):
    def __init__(self, effector, **kwargs):
        super().__init__(effector, max_ep_duration=5.0, **kwargs)
        self.device = effector.device
        # Task parameters
        self.goal_radius = 0.01  # 1cm target radius
        self.cue_delay = 0.2     # 200ms before movement allowed
        self.hold_threshold = int(0.8 / self.dt)  # 800ms in timesteps
        
        # Directions (8 evenly spaced angles)
        self.directions = th.linspace(0, 2*math.pi, 8, endpoint=False)
        
        # State tracking
        self.current_time = 0.0
        self.hold_counter = None
        self.goal_position = None
        self.center_position = None

    def reset(self, seed=None, options=None):
        options = options or {}
        batch_size = options.get("batch_size", 1)
        
        # 1. Reset to center position
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. Get center position from initial state
        self.center_position = self.effector.joint2cartesian(
            self.effector.joint_states.q
        )
        
        # 3. Randomly select directions for each batch element
        dir_idx = th.randint(0, 8, (batch_size,))
        angles = self.directions[dir_idx]
        
        # 4. Calculate target positions (center + direction vector)
        dx = self.goal_radius * th.cos(angles)
        dy = self.goal_radius * th.sin(angles)
        self.goal_position = self.center_position + th.stack([dx, dy], dim=-1)
        
        # 5. Initialize counters
        self.current_time = 0.0
        self.hold_counter = th.zeros(batch_size, device=self.device)
        info["goal"] = self.goal_position

        return obs, info

    def step(self, action, deterministic=False):
        # Freeze actions during cue delay
        if self.current_time < self.cue_delay:
            action = th.zeros_like(action)
            
        # Perform environment step
        obs, reward, terminated, truncated, info = super().step(
            action, deterministic=deterministic
        )
        
        # Update timers
        self.current_time += self.dt
        
        # Calculate target proximity
        fingertip_pos = info["states"]["fingertip"]
        dist_to_target = th.norm(fingertip_pos - self.goal_position, dim=-1)
        in_target = dist_to_target < self.goal_radius
        
        # Update hold counter and check termination
        self.hold_counter = th.where(in_target, self.hold_counter + 1, 0)
        terminated = terminated | (self.hold_counter >= self.hold_threshold)
        
        return obs, reward, terminated, truncated, info

    def get_obs(self, action=None, deterministic=False):
        # Add go-cue signal to observations
        base_obs = super().get_obs(action, deterministic)
        cue_signal = (self.current_time >= self.cue_delay).float().unsqueeze(-1)
        return th.cat([base_obs, cue_signal], dim=-1)

def create_defaultReachTask(effector):

    return mn.environment.RandomTargetReach(effector=effector, max_ep_duration=5.0)


