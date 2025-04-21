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


    '''
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
    
    '''
    def reset(self, seed=None, options=None):
        options = options or {}
        batch_size = options.get("batch_size", 1)
        
        # Get position bounds only (exclude velocity bounds)
        pos_lb = self.effector.pos_lower_bound
        pos_ub = self.effector.pos_upper_bound
        
        # Convert to tensors if needed
        if isinstance(pos_lb, np.ndarray):
            pos_lb = th.tensor(pos_lb, device=self.device, dtype=th.float32)
        if isinstance(pos_ub, np.ndarray):
            pos_ub = th.tensor(pos_ub, device=self.device, dtype=th.float32)
        
        # Generate random TARGET positions within joint limits
        q_pos_target = th.rand(batch_size, pos_lb.numel(), device=self.device) * (pos_ub - pos_lb) + pos_lb
        
        # Create full joint state (positions + zero velocities)
        q_target = th.cat([
            q_pos_target,
            th.zeros((batch_size, pos_lb.numel()), device=self.device)  # Zero velocities
        ], dim=-1)
        
        # Create START position 1cm from target (Cartesian space)
        cart_target = self.effector.joint2cartesian(q_target)
        cart_start = cart_target + th.randn_like(cart_target) * 0.01  # ~1cm noise
        
        # Find joint angles for start position (with velocities)
        q_start = th.cat([
            q_pos_target + th.randn_like(q_pos_target) * 0.01,  # Perturbed positions
            th.zeros_like(q_pos_target)  # Zero velocities
        ], dim=-1)

        # Reset environment with modified positions
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)
        
        # Store task parameters
        self.goal_position = cart_target
        self.current_time = 0.0
        self.hold_counter = th.zeros(batch_size, device=self.device)
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

class RandomTargetReach(mn.environment.Environment):
    """A custom reaching task:
       - The effector starts at a random state.
       - The target is drawn uniformly from the joint space (projected to Cartesian space),
         with a 1 cm radius (r = 0.01 m).
       - A 200 ms go-cue delay is imposed during which no movement is allowed.
       - Episodes last for 5 seconds (max_ep_duration=5.0 s).
       - The episode terminates early if the effector's endpoint stays within the target 
         region for at least 800 ms.
       - The reward is given by: 
           Rₗ = - y_pos * L1_norm(xₜ - xₜ′) - y_ctrl * ( (uₜ * f / ∥f∥₂²)² ),
         with
           y_pos = 0 if ∥xₜ - xₜ′∥₂ < r, else 1, and
           y_ctrl = 1 if ∥xₜ - xₜ′∥₂ < r, else 0.03.
    """
    
    def __init__(self, effector, **kwargs):
        # Set task-specific parameters:
        self.distance_criteria = 0.005 # 0.5cm radius from target
        self.target_radius = 0.01       # 1 cm radius target in meters
        self.cue_delay = 0.2            # 200 ms no-move phase
        # We assume the environment's dt is defined (e.g. dt = 0.02 s)
        self.dt = kwargs.pop("dt", 0.02)  # default timestep
        self.hold_threshold = 0.8       # 800 ms hold threshold (in seconds)
        self.elapsed = 0.0              # elapsed time in the episode
        self.hold_time = 0.0            # duration the effector is continuously within target

        # Pass max_ep_duration to parent (5 seconds)
        kwargs.setdefault("max_ep_duration", 5.0)
        super().__init__(effector, **kwargs)
        self.__name__ = "RandomTargetReach"
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Reset the environment:
          - Draw a random target joint state.
          - Define a start joint state that is 1 cm away from the target (by adding small random noise).
          - Convert joint states to Cartesian coordinates to set the target.
          - Reset internal timers and observation buffers.
        """
        self._set_generator(seed)  # set PRNG seeds
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        deterministic: bool = options.get('deterministic', False)

        # 1. Sample random start joint state from the full joint space.
        q_target = self.effector.draw_random_uniform_states(batch_size)  # shape: (batch_size, n_joints)
        q_start = self.effector.draw_random_uniform_states(batch_size)

        # 4. Reset the effector using the start joint state.
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)
        
        # 4. Set the goal.
        cart_goal = self.joint2cartesian(q_target)
        self.goal = cart_goal if self.differentiable else self.detach(cart_goal)
        info["goal"] = cart_goal
        
        # 6. Reset internal timers.
        self.elapsed = 0.0
        self.hold_time = 0.0

        # 7. Initialize observation buffers.
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        # 8. Get the initial observation.
        obs = self.get_obs(deterministic=deterministic)
        info.update({
        "states": self.states,
        "action": action,
        "noisy action": action,  # no noise at reset
        })
        return obs, info

    def apply_noise(self, loc, noise):
        """
        Override the default noise application to disable noise.
        This method returns the input `loc` unchanged.
        """
        return loc
    
    def step(self, action: th.Tensor, deterministic: bool = False) -> tuple:
        """
        Perform one simulation step.
        
        - During the first 200ms (cue_delay), actions are suppressed (set to zero) so that the effector stays
          at the starting position.
        - After stepping the environment, update the elapsed time.
        - Track the time the effector's endpoint (fingertip) remains within the target.
        - Terminate early if the hold time exceeds the threshold (800ms).
        - Compute the reward according to:
             Rₗ = - y_pos * L1_norm(xₜ - xₜ′) - y_ctrl * ( (uₜ * f / ∥f∥₂²)² ),
          where:
             y_pos = 0 if ∥xₜ - xₜ′∥₂ < r, else 1
             y_ctrl = 1 if ∥xₜ - xₜ′∥₂ < r, else 0.03.
        """
        # Freeze actions during the cue delay.
        if self.elapsed < self.cue_delay:
            action = th.zeros_like(action)
        
        # Step the simulation using the parent method.
        obs, _, terminated, truncated, info = super().step(action, deterministic=deterministic)
        #?# self.elapsed += self.dt
        
        # Compute the distance between the fingertip and the goal.
        # Assume the effector state info includes "states" with key "fingertip".
        fingertip = info["states"]["fingertip"]
        #print(f'This is fingertip: {fingertip} and goal: {self.goal}')
        dist = th.norm(fingertip - self.goal[:, :2], dim=-1)  # L2 norm per batch
        
        # Update hold time: if every element in batch is within the target radius, increment hold_time.
        # (You can modify this if handling batches differently.)
        if (dist < self.distance_criteria).all():
            self.hold_time += self.dt
        else:
            self.hold_time = 0.0
        
        # Terminate early if hold_time exceeds the threshold (0.8 seconds).
        if self.hold_time >= self.hold_threshold:
            terminated = True
        
        # Compute reward:
        # Define y_pos and y_ctrl based on the distance.
        y_pos = th.where(dist < self.distance_criteria, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(dist < self.distance_criteria, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
        
        # L1 norm between current position (xₜ) and goal (xₜ′):
        pos_error = th.sum(th.abs(fingertip - self.goal[:, :2]), dim=-1)
        
        # For the control term, assume f (maximum isometric contraction vector) is provided by the effector,
        # or use ones as a default. Adjust normalization as needed.
        f = th.tensor(self.effector.tobuild__muscle['max_isometric_force'], dtype=th.float32)
        norm_f_squared = th.norm(f.clone().detach(), p=2) ** 2
        f_expanded = f.expand_as(action)
        # Compute the control term: square of (uₜ * f / norm_f_squared)
        ctrl_term = th.sum((action * f_expanded / norm_f_squared) ** 2, dim=-1)
        
        # Compute reward as described:
        reward = - y_pos * pos_error - y_ctrl * ctrl_term
        reward = reward.unsqueeze(-1)  # ensure shape is (batch_size, 1)
        
        # Optionally, add reward components to info for debugging:
        info["reward_components"] = {"pos_error": pos_error, "ctrl_term": ctrl_term}
        
        return obs, reward, terminated, truncated, info

    def get_q_start(self, q_target: th.Tensor) -> th.Tensor:
        """
        Generate a joint state (q_start) exactly 1 cm from q_target.
        Resamples uniformly until the distance is between 0.95 and 1.05 cm.

        Args:
            q_target (torch.Tensor): Tensor of shape (batch_size, n_joints)

        Returns:
            q_start (torch.Tensor): Tensor of shape (batch_size, n_joints)
        """
        batch_size = q_target.shape[0]
        q_start = self.effector.draw_random_uniform_states(batch_size)

        # Compute distance and resample until within 0.95–1.05 cm
        def compute_distance(q1, q2):
            return th.norm(q1 - q2, dim=1)

        dist = compute_distance(q_start[:2], q_target[:2])
        retries = 0
        max_retries = 10000

        while th.any((dist < 0.0095) | (dist > 0.0105)):
            mask = (dist < 0.0095) | (dist > 0.0105)
            if retries >= max_retries:
                print("⚠️ Max retries reached in get_q_start. Proceeding with closest match.")
                break

            # Resample only the invalid ones
            new_samples = self.effector.draw_random_uniform_states(mask.sum())
            q_start[mask] = new_samples
            dist = compute_distance(q_start, q_target)
            retries += 1

        return q_start



def create_defaultReachTask(effector):

    return mn.environment.RandomTargetReach(effector=effector, max_ep_duration=5.0)


