
import numpy as np

from simulator.dynamics.forces.gravity import Gravity

class Dynamics:
    def __init__(self):
        self.forces = [
            Gravity(),
        ]

    def differentiate(self, t: float, positions: np.ndarray, velocities: np.ndarray, props: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

        N = positions.shape[0]
        
        masses = np.broadcast_to((props["masses"]).reshape(-1, 1), (N, 3))
        
        resultants = sum([
            model.compute_force(t, positions, velocities, props) for model in self.forces
        ])
        
        accelerations = resultants/masses
        
        return (velocities, accelerations)
        
