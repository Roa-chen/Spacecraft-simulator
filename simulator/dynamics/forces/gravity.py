
import numpy as np

from simulator.dynamics.forces.ForceModel import ForceModel
from simulator.utils.constants import G
from simulator.environment.environment import Environment
from simulator.environment.object import Object


class Gravity(ForceModel):
    def __init__(self):
        super().__init__()
        self.gravitational_constant = G

    def compute_force(self, t: float, positions: np.ndarray, velocities: np.ndarray, props: dict[str, np.ndarray]) -> np.ndarray:

        N = positions.shape[0]
        
        # Calculate distances 
        pos1 = positions.reshape(N, 1, 3)
        pos2 = positions.reshape(1, N, 3)
        dist = pos2 - pos1
        
        # Handle division by 0
        I = np.eye(N, dtype=float)
        I3 = np.repeat(I[:, :, np.newaxis], 3, axis=2)
        
        dist += I3
        
        # Calculate G*m_i*m_j
        
        masses = props["masses"].reshape(-1, 1)
        masses_t = np.transpose(masses)
        coefficients = G * masses * masses_t
        
        coefficients3 = np.repeat(coefficients[:, :, np.newaxis], 3, axis=2)
        
        # Unify
        
        D = np.repeat((np.linalg.norm(dist, axis=2)**3)[:, :, np.newaxis], 3, axis=2)
        
        forces3 = coefficients3*dist / D
        mask = np.ones((N, N)) - I
        forces3 *= (np.repeat(mask[:, :, np.newaxis], 3, axis=2))
        forces = np.sum(forces3, axis=1)
        
        return forces