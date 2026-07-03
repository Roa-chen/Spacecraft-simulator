
import numpy as np

from simulator.utils.quaternion import body_to_inertial
from simulator.dynamics.actions.actionModel import ActionModel
from simulator.dynamics.actions.gravity import Gravity
from simulator.actuators.actuatorModel import ActuatorModelManager
from simulator.actuators.reactionWheel import ReactionWheelManager

class Dynamics:
    considered_actions: list[ActionModel] = [
        Gravity,
    ]
        
    actuators: list[ActuatorModelManager] = [
        ReactionWheelManager,
    ]
    
    @staticmethod
    def differentiate(t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:

        N = state_props["MASS"].shape[0]
        Y = state.shape[0]
        
        d_state = np.zeros(Y)
        
        #Compute external actions
        
        screws_ext = [action.compute_action(t, state, state_indices, state_props) for action in Dynamics.considered_actions]
        resultant_ext = sum([screw[0] for screw in screws_ext])
        torque_ext = sum([screw[1] for screw in screws_ext])
        
        # Compute internal actions
        
        active_actuators = [actuator for actuator in Dynamics.actuators if actuator.is_in_state_vector(state_indices, state_props)]

        screws_int_body_frame = [actuator.get_action_in_body_frame(t, state, state_indices, state_props) for actuator in active_actuators]
        
        resultant_int_body_frame = sum([screw[0] for screw in screws_int_body_frame]) if active_actuators else np.zeros((N, 3))
        resultant_int = body_to_inertial(resultant_int_body_frame, state[state_indices["ATTITUDE"]].reshape(-1, 4)) #TODO: verify that this is correct
        
        torque_int = sum([screw[1] for screw in screws_int_body_frame]) if active_actuators else np.zeros((N, 3))
        
        # Combine external and internal actions
        
        resultant = resultant_ext + resultant_int
        torque = torque_ext + torque_int
        
        
        # Handle POSITION
        
        d_state[state_indices["POSITION"]] = state[state_indices["VELOCITY"]]
        
        # Handle VELOCITY
        
        masses = np.broadcast_to((state_props["MASS"]).reshape(-1, 1), (N, 3))
        accelerations = resultant/masses
        
        d_state[state_indices["VELOCITY"]] = accelerations.flatten()
        
        # Handle actuators
        
        for actuator in active_actuators:
            d_state += actuator.differentiate(t, state, state_indices, state_props)
        
        # Handle ATTITUDE
        
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
        
        # Handle ANGULAR_VELOCITY
        
        I = state_props["INERTIA_MATRIX"]
        I_inv = state_props["INERTIA_MATRIX_INV"]
        
        Iw = np.einsum('nij,nj->ni', I, angular_velocity)
        cross = np.cross(angular_velocity, Iw, axis=-1)
        
        d_angular_velocity = np.einsum('nij,nj->ni', I_inv, torque - cross)
        
        d_state[state_indices["ANGULAR_VELOCITY"]] = d_angular_velocity.flatten()

        return d_state
        
