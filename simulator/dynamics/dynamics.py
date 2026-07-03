
import numpy as np

from simulator.dynamics.actions.gravity import Gravity

class Dynamics:
    def __init__(self):
        self.forces = [
            Gravity(),
        ]
        
    """
    props are:
    MASS -> mass
    INERTIA ->
    INERTIA_INV -> 
    """    

    def differentiate(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:

        N = state_props["MASS"].shape[0]
        Y = state.shape[0]
        
        d_state = np.zeros(Y)
        
        screws = [model.compute_action(t, state, state_indices, state_props) for model in self.forces]
        
        # Handle POSITION
        
        d_state[state_indices["POSITION"]] = state[state_indices["VELOCITY"]]
        
        # Handle VELOCITY
        
        masses = np.broadcast_to((state_props["MASS"]).reshape(-1, 1), (N, 3))
        resultants = sum([screw[0] for screw in screws])
        accelerations = resultants/masses
        
        d_state[state_indices["VELOCITY"]] = accelerations.flatten()
        
        # Handle ATTITUDE
        
        K = state_props["SPACECRAFT_INDICES"].shape[0]
        
        attitude = (state[state_indices["ATTITUDE"]].reshape(N, 4))[state_props["SPACECRAFT_INDICES"]]
        angular_velocity = (state[state_indices["ANGULAR_VELOCITY"]].reshape(N, 3))[state_props["SPACECRAFT_INDICES"]]
        
        Omega = np.zeros((K, 4, 4))
        Omega[:, 1, 0] = angular_velocity[:, 0]
        Omega[:, 2, 0] = angular_velocity[:, 1]
        Omega[:, 3, 0] = angular_velocity[:, 2]
        
        Omega[:, 0, 1] = - angular_velocity[:, 0]
        Omega[:, 2, 1] = - angular_velocity[:, 2]
        Omega[:, 3, 1] = angular_velocity[:, 1]
        
        Omega[:, 0, 2] = - angular_velocity[:, 1]
        Omega[:, 1, 2] = angular_velocity[:, 2]
        Omega[:, 3, 2] = - angular_velocity[:, 0]
        
        Omega[:, 0, 3] = - angular_velocity[:, 2]
        Omega[:, 1, 3] = - angular_velocity[:, 1]
        Omega[:, 2, 3] = angular_velocity[:, 0]
        
        d_attitude = np.zeros((N, 4))
        d_attitude[state_props["SPACECRAFT_INDICES"]] = .5 * np.einsum('nij,nj->ni', Omega, attitude)
        
        d_state[state_indices["ATTITUDE"]] = d_attitude.flatten()
        
        # Handle ANGULAR_VELOCITY
        
        I = state_props["INERTIA_MATRIX"]
        I_inv = state_props["INERTIA_MATRIX_INV"]
        
        Iw = np.einsum('nij,nj->ni', I, angular_velocity)
        cross = np.cross(angular_velocity, Iw, axis=-1)   
        torque = sum([screw[1] for screw in screws])[state_props["SPACECRAFT_INDICES"]]
        
        d_angular_velocity = np.zeros((N, 3))
        d_angular_velocity[state_props["SPACECRAFT_INDICES"]] = np.einsum('nij,nj->ni', I_inv, torque - cross)
        
        d_state[state_indices["ANGULAR_VELOCITY"]] = d_angular_velocity.flatten()

        return d_state
        
