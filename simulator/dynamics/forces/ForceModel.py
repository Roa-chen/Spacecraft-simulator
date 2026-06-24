
import numpy as np
from simulator.environment.environment import Environment
from simulator.environment.object import Object

class ForceModel:

    def __init__(self):
        pass

    def compute_force(self, obj: Object, env: Environment) -> np.ndarray:
        pass