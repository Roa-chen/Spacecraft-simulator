

from simulator.environment.environment import Environment
from simulator.dynamics.dynamics import Dynamics
from simulator.core.clock import Clock


class Simulation:
    def __init__(self, environment: Environment, dynamics: Dynamics, dt: float):
        self.env = environment
        self.dynamics = dynamics
        self.dt = dt

        self.clock = Clock()

    def step(self):

        self.dynamics.step(self.env, self.dt)
        self.clock.step(self.dt)