
import numpy as np

class ActionModel:

    @staticmethod
    def compute_action(t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()