import numpy as np

from simulator.core.state import State
from simulator.environment.object import Object
from simulator.dynamics.integrators.integrator import Integrator

class Euler(Integrator):
    def __init__(self):
        super().__init__()

    def step(self, obj: Object, force: np.ndarray, dt: float):

        a_vec = force / obj.mass
        obj.state.velocity += a_vec
        obj.state.position += obj.state.velocity * dt

        