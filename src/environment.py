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
        self.batch_size = kwargs.pop("batch_size", 64)
        self.hold_threshold = 0.8       # 800 ms hold threshold (in seconds)
        self.elapsed = 0.0              # elapsed time in the episode
        self.hold_time = th.zeros(self.batch_size, device=self.device)            # duration the effector is continuously within target

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
        self.hold_time = th.zeros(batch_size, device=self.device)

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
        self.elapsed += self.dt
        
        # Compute the distance between the fingertip and the goal.
        # Assume the effector state info includes "states" with key "fingertip".
        fingertip = info["states"]["fingertip"]
        dist = th.norm(fingertip - self.goal[:, :2], dim=-1)  # L2 norm per batch
        
        # Update hold time: increment where distance < threshold, reset otherwise
        in_target = dist < self.distance_criteria
        self.hold_time = th.where(in_target, self.hold_time + self.dt, th.zeros_like(self.hold_time))

        # Terminate where hold_time exceeds threshold
        terminated = self.hold_time >= self.hold_threshold
        truncated = th.zeros_like(terminated, dtype=th.bool)
        
        # Compute reward:
        # Compute reward components
        y_pos = th.where(in_target, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(in_target, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
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

class ObstacleReach(mn.environment.Environment):
    """A custom reaching task that punishes passing through a forbidden rectangle."""

    def __init__(self, effector, **kwargs):
            # Set task-specific parameters:
            self.distance_criteria = 0.005 # 0.5cm radius from target
            self.target_radius = 0.01       # 1 cm radius target in meters
            self.cue_delay = 0.2            # 200 ms no-move phase
            # We assume the environment's dt is defined (e.g. dt = 0.02 s)
            self.dt = kwargs.pop("dt", 0.02)  # default timestep
            self.batch_size = kwargs.pop("batch_size", 32)
            self.hold_threshold = 0.8       # 800 ms hold threshold (in seconds)
            self.elapsed = 0.0              # elapsed time in the episode
            self.hold_time = th.zeros(self.batch_size, device=self.device)            # duration the effector is continuously within target
            self.forbidden_rect = (-0.2, 0.3, 0.2, 0.4)  # xmin, ymin, xmax, ymax
            self.penalty = -10000.0  # Heavy penalty for passing through the rectangle
            self.prev_fingertip = None

            # Pass max_ep_duration to parent (5 seconds)
            kwargs.setdefault("max_ep_duration", 5.0)
            super().__init__(effector, **kwargs)
            self.__name__ = "ObstacleReach"

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

        # 1. Sample random start joint state for valid start and target points (not in rectangle).
        q_target = self.generate_valid_q(batch_size)
        q_start = self.generate_valid_q(batch_size)

        # 4. Reset the effector using the start joint state.
        options["joint_state"] = q_start
        obs, info = super().reset(seed=seed, options=options)
        
        # 4. Set the goal.
        cart_goal = self.joint2cartesian(q_target)
        self.goal = cart_goal if self.differentiable else self.detach(cart_goal)
        info["goal"] = cart_goal
        
        # 6. Reset internal timers.
        self.elapsed = 0.0
        self.hold_time = th.zeros(batch_size, device=self.device)

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

        #Store initial fingertip position
        self.prev_fingertip = info["states"]["fingertip"].clone()

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
        self.elapsed += self.dt

        # Compute intersection to forbidden rectangle
        rectangle_penalties = th.zeros_like(self.hold_time)
        current_fingertip = info["states"]["fingertip"]
        if self.prev_fingertip is not None:
            x0 = self.prev_fingertip[:, 0]
            y0 = self.prev_fingertip[:, 1]
            x1 = current_fingertip[:, 0]
            y1 = current_fingertip[:, 1]
            intersects = self.liang_barsky_batch(x0, y0, x1, y1, self.forbidden_rect)
            rectangle_penalties[intersects] += self.penalty
        rectangle_penalty = th.sum(rectangle_penalties, dim=-1)
        
        # Compute the distance between the fingertip and the goal.
        # Assume the effector state info includes "states" with key "fingertip".
        fingertip = info["states"]["fingertip"]
        dist = th.norm(fingertip - self.goal[:, :2], dim=-1)  # L2 norm per batch
        
        # Update hold time: increment where distance < threshold, reset otherwise
        in_target = dist < self.distance_criteria
        self.hold_time = th.where(in_target, self.hold_time + self.dt, th.zeros_like(self.hold_time))

        # Terminate where hold_time exceeds threshold
        terminated = self.hold_time >= self.hold_threshold
        truncated = th.zeros_like(terminated, dtype=th.bool)
        
        # Compute reward:
        # Compute reward components
        y_pos = th.where(in_target, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(in_target, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
        pos_error = th.sum(th.abs(fingertip - self.goal[:, :2]), dim=-1)
        
        # For the control term, assume f (maximum isometric contraction vector) is provided by the effector,
        # or use ones as a default. Adjust normalization as needed.
        f = th.tensor(self.effector.tobuild__muscle['max_isometric_force'], dtype=th.float32)
        norm_f_squared = th.norm(f.clone().detach(), p=2) ** 2
        f_expanded = f.expand_as(action)
        # Compute the control term: square of (uₜ * f / norm_f_squared)
        ctrl_term = th.sum((action * f_expanded / norm_f_squared) ** 2, dim=-1)
        
        # Compute reward as described:
        reward = - y_pos * pos_error - y_ctrl * ctrl_term - rectangle_penalty
        reward = reward.unsqueeze(-1)  # ensure shape is (batch_size, 1)
        
        # Optionally, add reward components to info for debugging:
        info["reward_components"] = {"pos_error": pos_error, "ctrl_term": ctrl_term}
        
        return obs, reward, terminated, truncated, info

    def generate_valid_q(self, batch_size):
        """Generates valid joint states uniformly while avoiding forbidden rectangle"""
        x_min, y_min, x_max, y_max = self.forbidden_rect
        
        # Get number of DOF from effector's joint state shape
        test_state = self.effector.draw_random_uniform_states(1)  # Shape: (1, n_dof)
        n_dof = test_state.shape[1]
        
        all_valid_q = th.zeros((0, n_dof), device=self.device)
        
        while all_valid_q.shape[0] < batch_size:
            n_needed = batch_size - all_valid_q.shape[0]
            candidates_q = self.effector.draw_random_uniform_states(n_needed)
            candidates_cart = self.joint2cartesian(candidates_q)
            
            # Check if inside forbidden rectangle
            x_in = (candidates_cart[:, 0] >= x_min) & (candidates_cart[:, 0] <= x_max)
            y_in = (candidates_cart[:, 1] >= y_min) & (candidates_cart[:, 1] <= y_max)
            valid_mask = ~(x_in & y_in)
            
            all_valid_q = th.cat([all_valid_q, candidates_q[valid_mask]], dim=0)
        
        return all_valid_q[:batch_size]

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
        # Set task-specific parameters:
        self.distance_criteria = 0.005 # 0.5cm radius from target
        self.target_radius = 0.1       # 10 cm radius target in meters
        self.cue_delay = 0.2            # 200 ms no-move phase
        # We assume the environment's dt is defined (e.g. dt = 0.02 s)
        self.dt = kwargs.pop("dt", 0.02)  # default timestep
        self.batch_size = kwargs.pop("batch_size", 32)
        self.hold_threshold = 0.5       # 500 ms hold threshold (in seconds)
        self.elapsed = 0.0              # elapsed time in the episode
        self.hold_time = th.zeros(self.batch_size, device=self.device)            # duration the effector is continuously within target
        self.directions = th.linspace( start=0,end=2 * math.pi * (1 - 1/8), steps=8, device=effector.device) # Directions (8 evenly spaced angles)
        # Pass max_ep_duration to parent (5 seconds)
        kwargs.setdefault("max_ep_duration", 1.0)
        super().__init__(effector, **kwargs)
        self.__name__ = "CenterOutReach"
    
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
        dx = self.target_radius * th.cos(angles)
        dy = self.target_radius * th.sin(angles)
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
        self.elapsed += self.dt
        
        # Compute the distance between the fingertip and the goal.
        # Assume the effector state info includes "states" with key "fingertip".
        fingertip = info["states"]["fingertip"]
        dist = th.norm(fingertip - self.goal[:, :2], dim=-1)  # L2 norm per batch
        
        # Update hold time: increment where distance < threshold, reset otherwise
        in_target = dist < self.distance_criteria
        self.hold_time = th.where(in_target, self.hold_time + self.dt, th.zeros_like(self.hold_time))

        # Terminate where hold_time exceeds threshold
        terminated = self.hold_time >= self.hold_threshold
        truncated = th.zeros_like(terminated, dtype=th.bool)
        
        # Compute reward:
        # Compute reward components
        y_pos = th.where(in_target, th.tensor(0.0, device=action.device), th.tensor(1.0, device=action.device))
        y_ctrl = th.where(in_target, th.tensor(1.0, device=action.device), th.tensor(0.03, device=action.device))
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

def create_defaultReachTask(effector):

    return mn.environment.RandomTargetReach(effector=effector, max_ep_duration=5.0)


