
import numpy as np

class Manager:
    @staticmethod
    def get_state_required_properties(count: dict[str, int]) -> tuple[dict[str, int], dict[str, np.shape]]:
        raise NotImplementedError("This method should be overridden by subclasses.")