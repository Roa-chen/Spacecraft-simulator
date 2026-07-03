from typing import TYPE_CHECKING

import numpy as np

from simulator.core.manager import Manager

if TYPE_CHECKING:
    from simulator.environment.spacecraft import Spacecraft

class ActuatorModel:
    def __init__(self):
        self.spacecraft: Spacecraft = None
    
    def initialize_count_manager(self, count: dict[str, int], manager: dict[str, Manager]):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def initialize_state(self, state: np.ndarray, state_indices: dict[str, slice], state_props: dict[str, np.ndarray], indices: dict[str, int]):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def apply(self, input_signal):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_action_in_body_frame(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def differentiate(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")