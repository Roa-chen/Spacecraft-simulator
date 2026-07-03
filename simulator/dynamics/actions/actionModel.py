
import numpy as np
from simulator.environment.environment import Environment
from simulator.environment.celestialObject import CelestialObject

class ActionModel:

    def __init__(self):
        pass

    """
    props are:
    MASS -> mass of the object
    """
    def compute_action(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()