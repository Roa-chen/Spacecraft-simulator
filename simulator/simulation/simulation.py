

from simulator.environment.environment import Environment
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.integrator import Integrator
from simulator.core.clock import Clock


class Simulation:
    def __init__(self, environment: Environment, integrator: Integrator, dt: float):
        self.env = environment
        self.integrator = integrator
        self.dt = dt

    def step(self):

        self.integrator.step(self.env, self.dt)
        self.env.step_time(self.dt)