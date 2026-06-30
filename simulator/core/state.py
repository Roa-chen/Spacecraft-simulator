import numpy as np

class State:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, attitude: np.ndarray, angular_velocity: np.ndarray) -> None:
        
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        self.attitude = np.array(attitude, dtype=float) # quaternion
        self.angular_velocity = np.array(angular_velocity, dtype=float) # Expressed in Body Frame