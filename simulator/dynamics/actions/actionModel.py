
import numpy as np
from simulator.environment.environment import Environment
from simulator.environment.object import Object

class ActionModel:

    def __init__(self):
        pass

    """
    props are:
    MASS -> mass of the object
    """
    def compute_action(self, t: float, positions: np.ndarray, velocities: np.ndarray, attitude: np.ndarray, angular_velocity: np.ndarray, props: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()