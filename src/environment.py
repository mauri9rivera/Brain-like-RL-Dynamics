import motornet as mn
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import math

class Arm:

    def __init__(self, arm_type, integrator='rk4', verbose=False):

        self.integrator = 'rk4'

        if arm_type == 'arm210':
            self.effector = self.build_arm210(verbose)
        elif arm_type == 'arm26':
            self.effector = self.build_arm26(verbose)

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

class CenterOutReachEnv(mn.environment.Environment):
    """
    A custom CenterOut Reaching task.
    - The effector starts at the origin.
    - A target is drawn from 8 possible directions
    - A 200 ms go-cue delay is imposed durhing which no movement is allows
    - Episodes last for 1 seconds (max_ep_duration=1.0 s)
    - The episode terminates early if the effector's endpoint stays within the target 
         region for at least 500 ms.
       - The reward is given by the L1 norm from the goal and the target
    """
    def __init__(self, effector, **kwargs):
        # Task parameters
        self.distance_criteria = 0.005 # 0.5cm radius from target
        self.goal_radius = 0.08  # 1cm target radius
        self.cue_delay = 0.2     # 200ms before movement allowed
        self.dt = kwargs.pop("dt", 0.02)  # default timestep
        self.elapsed = 0.0 # elapsed time in the episode
        self.hold_time = 0.0
        self.hold_threshold = 0.5  # 800ms in timesteps
        self.directions = th.linspace( start=0,end=2 * math.pi * (1 - 1/8), steps=8, device=effector.device) # Directions (8 evenly spaced angles)
        super().__init__(effector, max_ep_duration=5.0, **kwargs)
        
                
    
    def reset(self, seed=None, options=None):
        self._set_generator(seed)  # set PRNG seeds
        options = options or {}
        batch_size = options.get("batch_size", 1)
        deterministic: bool = options.get('deterministic', False)
        
        # 1. Reset to DEFAULT joint position (don't override joint_state)
        joint_states = self.effector.draw_fixed_states(batch_size, th.tensor([1.3, 1.4], dtype=th.float32))
        options['joint_state'] = joint_states
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. Get DEFAULT start position from actual fingertip location
        start_pos = info["states"]["fingertip"][:, :2]  # Shape: (batch, 2)
        
        # 3. Randomly select directions
        dir_idx = th.randint(0, 8, (batch_size,), device=self.device)
        angles = self.directions[dir_idx]
        
        # 4. Calculate targets RELATIVE TO ACTUAL START POSITION
        dx = self.goal_radius * th.cos(angles)
        dy = self.goal_radius * th.sin(angles)
        self.goal = start_pos + th.stack([dx, dy], dim=-1)  # Save as instance var

        # 5. Update info and timers
        info["goal"] = self.goal
        self.elapsed = 0.0
        self.hold_time = th.zeros(batch_size, device=self.device)

        # 6. Initialize observation buffers.
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        # 7. Get the initial observation.
        obs = self.get_obs(deterministic=deterministic)
        info.update({
        "states": self.states,
        "action": action,
        "noisy action": action,  # no noise at reset
        })

        obs = self.to_numpy(obs)
        reward = reward.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated if isinstance(truncated, bool) else truncated.detach().cpu().numpy()

        return obs, reward, terminated.all(), truncated, info

    def to_numpy(self, x):
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
        return x

    def apply_noise(self, loc, noise):
        """
        Override the default noise application to disable noise.
        This method returns the input `loc` unchanged.
        """
        return loc

    def step(self, action, deterministic=False):
        """
        Perform one simulation step.
        
        - During the first 200ms (cue_delay), actions are suppressed (set to zero) so that the effector stays
          at the starting position.
        - After stepping the environment, update the elapsed time.
        - Track the time the effector's endpoint (fingertip) remains within the target.
        - Terminate early if the hold time exceeds the threshold (500ms).
        - Compute the reward according to the L1 norm from the goal
        """
        if not isinstance(action, th.Tensor):
            action = th.as_tensor(action, device=self.device, dtype=th.float32)
        else:
            action = action.clone().detach().to(device=self.device, dtype=th.float32)
        # Freeze actions during cue delay
        if self.elapsed < self.cue_delay:
            action = th.zeros_like(action)
            
        # Perform environment step
        obs, _, terminated, truncated, info = super().step(action, deterministic=deterministic)
        self.elapsed += self.dt
        
        # Calculate distance to TARGET (using saved self.goal)
        fingertip_pos = info["states"]["fingertip"][:, :2]
        dist = th.norm(fingertip_pos - self.goal, dim=-1)

        # Update hold time individually for each element
        within_target = dist < self.distance_criteria
        self.hold_time = th.where(within_target,
                                self.hold_time + self.dt,
                                th.zeros_like(self.hold_time))
        
        # Terminate early if hold_time exceeds the threshold (0.8 seconds).
        terminated = self.hold_time >= self.hold_threshold

        #Compute reward as described:
        reward = -dist.unsqueeze(-1) # ensure shape is (batch_size, 1)
        
        return obs, reward, terminated.all(), truncated, info
    '''
    def get_obs(self, action=None, deterministic=False):
        # Add go-cue signal to observations
        base_obs = super().get_obs(action, deterministic)
        cue_signal = (self.current_time >= self.cue_delay).float().unsqueeze(-1)
        return th.cat([base_obs, cue_signal], dim=-1)
    '''

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
        self.target_radius = 1.0      # 1 cm radius target in meters
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

        # 1. Sample a random start joint state from the full joint space.
        q_start = self.effector.draw_random_uniform_states(batch_size)  # shape: (batch_size, n_joints)
        
        # Convert joint state to Cartesian coordinates.
        cart_start = self.joint2cartesian(q_start)

        # 2. Compute a random 1 cm offset in Cartesian space.
        # Generate a random direction vector with the same shape as cart_start.
        rand_direction = th.randn_like(cart_start[:2])
        # Normalize the direction vector (for each sample).
        norm = th.norm(rand_direction, dim=1, keepdim=True)
        # Avoid division by zero.
        norm = th.where(norm < 1e-6, th.ones_like(norm), norm)
        unit_direction = rand_direction / norm

        # Scale the unit direction to exactly 1 cm.
        offset = self.target_radius * unit_direction
        cart_goal = cart_start
        cart_goal[:2] = cart_start[:2] + offset

        # 4. Reset the effector using the start joint state.
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)
        
        # 5. Set the goal.
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
        if (dist < self.target_radius).all():
            self.hold_time += self.dt
        else:
            self.hold_time = 0.0
        
        # Terminate early if hold_time exceeds the threshold (0.8 seconds).
        if self.hold_time >= self.hold_threshold:
            terminated = True
        
        # Compute reward:
        # Define y_pos and y_ctrl based on the distance.
        y_pos = th.where(dist < self.target_radius, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(dist < self.target_radius, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
        
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

class RandomTargetReach2(mn.environment.Environment):
    """A custom reaching task:
       - The effector starts  n a random state drawn uniformly from the joint space 
       - The target is drawn uniformly at a state 1 cm away from the origin in Cartesian space.
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
        self.target_radius = 0.01     # 1 cm radius target 
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
          - Draw a random target joint state as starting position
          - Define a target that is 1 cm away from the origin.
          - Reset internal timers and observation buffers.
        """
        self._set_generator(seed)  # set PRNG seeds
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        deterministic: bool = options.get('deterministic', False)

        # 1. Generate q_target as a random position 1cm from the origin TODO
        '''
        rand_direction = th.randn((batch_size, 2), device=self.device)
        norm = th.norm(rand_direction, dim=1, keepdim=True)
        norm = th.where(norm < 1e-6, th.ones_like(norm), norm)
        unit_direction = rand_direction / norm
        rand_pos = self.target_radius * unit_direction
        rand_pos = th.clamp(
            rand_pos,
            min=th.tensor(self.effector.pos_lower_bound[None, :]),  
            max=th.tensor(self.effector.pos_upper_bound[None, :])) # Ensure positions are within effector's workspace bounds
        q_target = th.stack([
            self.effector.draw_fixed_states(1, pos.unsqueeze(0))
            for pos in rand_pos], dim=0).squeeze(1)
        #q_target = self.effector.draw_fixed_states(batch_size, rand_pos)
        ''' 
        q_target = self.effector.draw_random_uniform_states(batch_size)

        # 2. Generate q_start drawn uniformly from joint space
        q_start = self.effector.draw_random_uniform_states(batch_size)

        # 3. Reset the effector using the start joint state.
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)
        
        # 4. Set the goal.
        cart_goal = self.joint2cartesian(q_target)
        self.goal = cart_goal if self.differentiable else self.detach(cart_goal)
        info["goal"] = cart_goal
        
        # 5. Reset internal timers.
        self.elapsed = 0.0
        self.hold_time = th.zeros(batch_size, device=self.device)

        # 6. Initialize observation buffers.
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        # 7. Get the initial observation.
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
    
    def step(self, action, deterministic: bool = False) -> tuple:
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
        if not isinstance(action, th.Tensor):
            action = th.as_tensor(action, device=self.device, dtype=th.float32)
        else:
            action = action.clone().detach().to(device=self.device, dtype=th.float32)

        # Freeze actions during the cue delay.
        if self.elapsed < self.cue_delay:
            action = th.zeros_like(action)
        
        # Step the simulation using the parent method.
        obs, _, terminated, truncated, info = super().step(action, deterministic=deterministic)
        self.elapsed += self.dt
        
        # Compute the distance between the fingertip and the goal.
        fingertip = info["states"]["fingertip"]
        dist = th.norm(fingertip - self.goal[:, :2], dim=-1)  # L2 norm per batch
        
        # Update hold time individually for each element
        within_target = dist < self.distance_criteria
        self.hold_time = th.where(within_target,
                                self.hold_time + self.dt,
                                th.zeros_like(self.hold_time))
            
        # Terminate early if hold_time exceeds the threshold (0.8 seconds).
        terminated = self.hold_time >= self.hold_threshold
        
        # Compute reward:
        # Define y_pos and y_ctrl based on the distance.
        y_pos = th.where(within_target, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(within_target, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
        
        # L1 norm between current position (xₜ) and goal (xₜ′):
        pos_error = th.sum(th.abs(fingertip - self.goal[:, :2]), dim=-1)
        
        # Assume f is maximum isometric contraction vector
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
        
        obs = self.to_numpy(obs)
        reward = reward.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated if isinstance(truncated, bool) else truncated.detach().cpu().numpy()

        return obs, reward, terminated.all(), truncated, info

    def to_numpy(self, x):
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
        return x

class ForbiddenRectangleReach(RandomTargetReach2):
    """A custom reaching task that punishes passing through a forbidden rectangle."""
    
    def __init__(self, effector, **kwargs):
        self.forbidden_rect = (-0.2, 0.3, 0.2, 0.4)  # xmin, ymin, xmax, ymax
        self.penalty = -10000.0  # Heavy penalty for passing through the rectangle
        self.prev_fingertip = None
        self.__name__ = "ForbiddenRectangleReach"
        super().__init__(effector, **kwargs)
        
        

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        deterministic = options.get('deterministic', False)

        q_target = self.generate_valid_q(batch_size, is_target=True)
        q_start = self.generate_valid_q(batch_size, is_target=False)

        # Reset effector with new q_start
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)

        # Update goal to new q_target
        cart_goal = self.joint2cartesian(q_target)
        self.goal = cart_goal if self.differentiable else self.detach(cart_goal)
        info["goal"] = cart_goal

        # Store initial fingertip position
        self.prev_fingertip = info["states"]["fingertip"].clone()

        return obs, info

    def step(self, action, deterministic: bool = False) -> tuple:
        obs, reward, terminated, truncated, info = super().step(action, deterministic)

        current_fingertip = info["states"]["fingertip"]
        if self.prev_fingertip is not None:
            x0 = self.prev_fingertip[:, 0]
            y0 = self.prev_fingertip[:, 1]
            x1 = current_fingertip[:, 0]
            y1 = current_fingertip[:, 1]
            intersects = self.liang_barsky_batch(x0, y0, x1, y1, self.forbidden_rect)
            reward[intersects] += self.penalty

        self.prev_fingertip = current_fingertip.clone()

        return obs, reward, terminated, truncated, info

    @staticmethod
    def liang_barsky_batch(x0, y0, x1, y1, rect):
        dx = x1 - x0
        dy = y1 - y0
        p = th.stack([-dx, dx, -dy, dy], dim=1)
        q = th.stack([x0 - rect[0], rect[2] - x0, y0 - rect[1], rect[3] - y0], dim=1)

        t0 = th.zeros_like(x0)
        t1 = th.ones_like(x0)

        for i in range(4):
            mask_p_ne_zero = p[:, i] != 0
            if not th.any(mask_p_ne_zero):
                continue
            t = q[:, i] / p[:, i]
            if i % 2 == 0:
                mask_p_neg = mask_p_ne_zero & (p[:, i] < 0)
                t0 = th.where(mask_p_neg, th.maximum(t0, t), t0)
                mask_p_pos = mask_p_ne_zero & (p[:, i] > 0)
                t1 = th.where(mask_p_pos, th.minimum(t1, t), t1)
            else:
                mask_p_neg = mask_p_ne_zero & (p[:, i] < 0)
                t1 = th.where(mask_p_neg, th.minimum(t1, t), t1)
                mask_p_pos = mask_p_ne_zero & (p[:, i] > 0)
                t0 = th.where(mask_p_pos, th.maximum(t0, t), t0)

        intersects = (t0 <= t1) & (t0 < 1.0) & (t1 > 0.0)
        return intersects

    def generate_valid_q(self, batch_size, is_target=False):
        """Generates valid joint states ensuring Cartesian positions are outside the forbidden rectangle."""
        if is_target:
            # Generate target positions 1cm from origin in Cartesian space, then convert to joint states
            rand_direction = th.randn((batch_size, 2), device=self.device)
            norm = th.norm(rand_direction, dim=1, keepdim=True)
            norm = th.where(norm < 1e-6, th.ones_like(norm), norm)
            unit_direction = rand_direction / norm
            rand_pos = self.target_radius * unit_direction
            rand_pos = th.clamp(rand_pos,
                                min=th.tensor(self.effector.pos_lower_bound, device=self.device),
                                max=th.tensor(self.effector.pos_upper_bound, device=self.device))
            
            # Reject positions inside forbidden area
            x_in = (rand_pos[:, 0] >= self.forbidden_rect[0]) & (rand_pos[:, 0] <= self.forbidden_rect[2])
            y_in = (rand_pos[:, 1] >= self.forbidden_rect[1]) & (rand_pos[:, 1] <= self.forbidden_rect[3])
            in_forbidden = x_in & y_in
            invalid_indices = th.where(in_forbidden)[0]

            while invalid_indices.numel() > 0:
                new_dir = th.randn(invalid_indices.size(0), 2, device=self.device)
                new_norm = th.norm(new_dir, dim=1, keepdim=True)
                new_norm = th.where(new_norm < 1e-6, th.ones_like(new_norm), new_norm)
                new_unit = new_dir / new_norm
                new_pos = self.target_radius * new_unit
                new_pos = th.clamp(new_pos,
                                min=th.tensor(self.effector.pos_lower_bound, device=self.device),
                                max=th.tensor(self.effector.pos_upper_bound, device=self.device))
                x_in_new = (new_pos[:, 0] >= self.forbidden_rect[0]) & (new_pos[:, 0] <= self.forbidden_rect[2])
                y_in_new = (new_pos[:, 1] >= self.forbidden_rect[1]) & (new_pos[:, 1] <= self.forbidden_rect[3])
                new_invalid = x_in_new & y_in_new
                valid = ~new_invalid
                valid_indices = invalid_indices[valid]
                if valid_indices.numel() > 0:
                    rand_pos[valid_indices] = new_pos[valid]
                invalid_indices = invalid_indices[new_invalid]

            q = th.stack([self.effector.draw_fixed_states(1, pos.unsqueeze(0)) for pos in rand_pos], dim=0).squeeze(1)
        else:
            # Generate joint states and check Cartesian positions
            q = self.effector.draw_random_uniform_states(batch_size)
            cart = self.joint2cartesian(q)
            x_in = (cart[:, 0] >= self.forbidden_rect[0]) & (cart[:, 0] <= self.forbidden_rect[2])
            y_in = (cart[:, 1] >= self.forbidden_rect[1]) & (cart[:, 1] <= self.forbidden_rect[3])
            in_forbidden = x_in & y_in
            invalid_indices = th.where(in_forbidden)[0]

            while invalid_indices.numel() > 0:
                new_q = self.effector.draw_random_uniform_states(invalid_indices.size(0))
                new_cart = self.joint2cartesian(new_q)
                x_in_new = (new_cart[:, 0] >= self.forbidden_rect[0]) & (new_cart[:, 0] <= self.forbidden_rect[2])
                y_in_new = (new_cart[:, 1] >= self.forbidden_rect[1]) & (new_cart[:, 1] <= self.forbidden_rect[3])
                new_invalid = x_in_new & y_in_new
                valid = ~new_invalid
                valid_indices_invalid = invalid_indices[valid]
                if valid_indices_invalid.numel() > 0:
                    q[valid_indices_invalid] = new_q[valid]
                    cart[valid_indices_invalid] = new_cart[valid]
                invalid_indices = invalid_indices[new_invalid]

        return q

def create_defaultReachTask(effector):

    return mn.environment.RandomTargetReach(effector=effector, max_ep_duration=5.0)


