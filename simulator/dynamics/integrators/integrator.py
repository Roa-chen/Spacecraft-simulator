
import numpy as np

from simulator.environment.object import Object
from simulator.environment.environment import Environment
from simulator.dynamics.dynamics import Dynamics

class Integrator():
    def __init__(self, dynamics: Dynamics):
        self.dynamics = dynamics

    def step(self, env: Environment, dt: float):
        pass