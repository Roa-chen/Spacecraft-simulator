
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

    def differentiate(self, t: float, positions: np.ndarray, velocities: np.ndarray, attitude: np.ndarray, angular_velocity: np.ndarray, props: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        N = positions.shape[0]
        
        screws = [model.compute_action(t, positions, velocities, attitude, angular_velocity, props) for model in self.forces]
        
        # Compute acceleration
        
        masses = np.broadcast_to((props["MASS"]).reshape(-1, 1), (N, 3))
        
        resultants = sum([screw[0] for screw in screws])
        
        accelerations = resultants/masses
        
        # Compute d_attitude
        
        Omega = np.zeros((N, 4, 4))
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
        
        d_attitude = .5 * np.einsum('nij,nj->ni', Omega, attitude)
        
        # Compute d_angular_velocity
        
        I = props['INERTIA']
        I_inv = props['INERTIA_INV']
        
        Iw = np.einsum('nij,nj->ni', I, angular_velocity)
        cross = np.cross(angular_velocity, Iw, axis=-1)
        
        torque = sum([screw[1] for screw in screws])
        
        d_angular_velocity = np.einsum('nij,nj->ni', I_inv, torque - cross)
        
        return (velocities, accelerations, d_attitude, d_angular_velocity)
        
