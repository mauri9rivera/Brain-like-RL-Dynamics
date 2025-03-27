import motornet as mn

class MotorNetEnv:
    def __init__(self):
        self.effector = mn.RigidTendonArm26()
        self.effector.set_integrator("RK4")  # Runge-Kutta 4
        self.actuators = [mn.MujocoHillMuscle() for _ in range(6)]  # No passive force

    def reset(self):
        return self.effector.reset()

    def step(self, action):
        self.effector.apply_action(action)
        state = self.effector.get_state()
        return state

    def render(self):
        self.effector.visualize()
