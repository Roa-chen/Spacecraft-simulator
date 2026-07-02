
import numpy as np

from simulator.dynamics.actions.actionModel import ActionModel
from simulator.utils.constants import G
from simulator.environment.environment import Environment
from simulator.environment.object import Object


class Gravity(ActionModel):
    def __init__(self):
        super().__init__()
        self.gravitational_constant = G

    def compute_action(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

        N = state_props["MASS"].shape[0]
        
        # Calculate distances 
        pos1 = state[state_indices["POSITION"]].reshape(N, 1, 3)
        pos2 = state[state_indices["POSITION"]].reshape(1, N, 3)
        dist = pos2 - pos1
        
        # Calculate 1/d^3
        
        denominator = np.linalg.norm(dist, axis=2)**3
        denominator += np.diag(np.inf*np.ones(N))
        
        # Calculate G*m_i*m_j / d^3
        
        mass = state_props["MASS"].reshape(-1, 1)
        mass_t = np.transpose(mass)
        coefficient = G * (mass * mass_t) * (1/denominator)
        
        # Unify
        
        force3 = np.einsum('ijk,ij->ijk', dist, coefficient)
        force = np.sum(force3, axis=1)
        
        return (force, np.zeros((N, 3)))