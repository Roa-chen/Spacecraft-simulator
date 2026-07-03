import numpy as np

class ActuatorModel:
    def __init__(self):
        self.spacecraft = None
        
    
    def apply(self, input_signal):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_action_in_body_frame(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def differentiate(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")