
import numpy as np

from simulator.dynamics.actions.actionModel import ActionModel
from simulator.dynamics.actions.gravity import Gravity
from simulator.actuators.actuatorModel import ActuatorModel

class Dynamics:
    def __init__(self):
        self.forces: list[ActionModel] = [
            Gravity(),
        ]
        
        self.actuators: list[ActuatorModel] = [
            
        ]
        
    """
    props are:
    MASS -> mass
    INERTIA ->
    INERTIA_INV -> 
    """    

    def differentiate(self, t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:

        N = state_props["MASS"].shape[0]
        K = state_props["SPACECRAFT_INDICES"].shape[0]
        Y = state.shape[0]
        
        d_state = np.zeros(Y)
        
        screws_ext = [model.compute_action(t, state, state_indices, state_props) for model in self.forces]
        resultant_ext = sum([screw[0] for screw in screws_ext])
        torque_ext = sum([screw[1] for screw in screws_ext])
        
        screws_int_body_frame = [actuator.get_action_in_body_frame(t, state, state_indices, state_props) for actuator in self.actuators]
        resultant_int_body_frame = sum([screw[0] for screw in screws_int_body_frame]) if self.actuators else np.zeros((N, 3))
        resultant_int = resultant_int_body_frame #FIXME: convert to inertial frame 
        torque_int = sum([screw[1] for screw in screws_int_body_frame]) if self.actuators else np.zeros((N, 3))
        
        resultant = resultant_ext + resultant_int
        torque = torque_ext + torque_int
        
        
        # Handle POSITION
        
        d_state[state_indices["POSITION"]] = state[state_indices["VELOCITY"]]
        
        # Handle VELOCITY
        
        masses = np.broadcast_to((state_props["MASS"]).reshape(-1, 1), (N, 3))
        accelerations = resultant/masses
        
        d_state[state_indices["VELOCITY"]] = accelerations.flatten()
        
        # Handle actuators
        
        for actuator in self.actuators:
            d_state += actuator.differentiate(t, state, state_indices, state_props)
        
        # Handle ATTITUDE
        
        K = state_props["SPACECRAFT_INDICES"].shape[0]
        
        attitude = (state[state_indices["ATTITUDE"]].reshape(N, 4))
        angular_velocity = (state[state_indices["ANGULAR_VELOCITY"]].reshape(N, 3))
        
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
        
        d_attitude = np.zeros((N, 4))
        d_attitude = .5 * np.einsum('nij,nj->ni', Omega, attitude)
        
        d_state[state_indices["ATTITUDE"]] = d_attitude.flatten()
        
        # Handle ANGULAR_VELOCITY (only spacecrafts)
        
        angular_velocity_spacecraft = angular_velocity[state_props["SPACECRAFT_INDICES"]]
        
        I = state_props["INERTIA_MATRIX"]
        I_inv = state_props["INERTIA_MATRIX_INV"]
        
        Iw = np.einsum('nij,nj->ni', I, angular_velocity_spacecraft)
        cross = np.cross(angular_velocity_spacecraft, Iw, axis=-1)
        
        torque_int_spacecraft = torque_int[state_props["SPACECRAFT_INDICES"]]
        
        d_angular_velocity = np.zeros((N, 3))
        d_angular_velocity_spacecraft = np.einsum('nij,nj->ni', I_inv, torque_int_spacecraft - cross)
        d_angular_velocity[state_props["SPACECRAFT_INDICES"]] = d_angular_velocity_spacecraft
        
        d_state[state_indices["ANGULAR_VELOCITY"]] = d_angular_velocity.flatten()

        return d_state
        
