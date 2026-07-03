from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.environment.spacecraft import Spacecraft

from simulator.core.manager import Manager
from simulator.actuators.actuatorModel import ActuatorModel

class reactionWheelManager(Manager):
    """
    KEY -> REACTION_WHEEL
    """
    
    @staticmethod
    def get_state_required_properties(count: dict[str, int]) -> tuple[dict[str, int], dict[str, np.shape]]:
        M = count.get("REACTION_WHEEL", 0)
        required_state = {
            "REACTION_WHEEL_ANGULAR_VELOCITY": M,
            "REACTION_WHEEL_TORQUE_CMD": M,
        }
        required_state_props = {
            "REACTION_WHEEL_ORIENTATION": (M, 3),
            "REACTION_WHEEL_INERTIA": M,
            "REACTION_WHEEL_MAX_TORQUE": M,
            "REACTION_WHEEL_MAX_VELOCITY": M,
            "REACTION_WHEEL_STATIC_FRICTION_COEFFICIENT": M,
            "REACTION_WHEEL_DYNAMIC_FRICTION_COEFFICIENT": M,
            "REACTION_WHEEL_SPACECRAFT_MATRIX": (count.get("SPACECRAFT", 0), M),
        }
        return required_state, required_state_props
    
    @staticmethod
    def get_action_in_body_frame(t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        
        N = state_props["MASS"].shape[0]
        
        reaction_wheel_torque_cmd = np.clip(state[state_indices["REACTION_WHEEL_TORQUE_CMD"]], -state_props["REACTION_WHEEL_MAX_TORQUE"], state_props["REACTION_WHEEL_MAX_TORQUE"])
        reaction_wheel_dynamic_friction_coefficient = state_props["REACTION_WHEEL_DYNAMIC_FRICTION_COEFFICIENT"]
        reaction_wheel_angular_velocity = state[state_indices["REACTION_WHEEL_ANGULAR_VELOCITY"]]
        
        reaction_wheel_torque = reaction_wheel_torque_cmd - reaction_wheel_dynamic_friction_coefficient * reaction_wheel_angular_velocity
        reaction_wheel_spacecraft_matrix = state_props["REACTION_WHEEL_SPACECRAFT_MATRIX"]
        
        reaction_wheel_torque_1D = np.matmul(reaction_wheel_spacecraft_matrix, reaction_wheel_torque) 
        
        spacecraft_torque = - np.einsum('mi,m->mi', state_props["REACTION_WHEEL_ORIENTATION"], reaction_wheel_torque_1D) # '-' signe because it is the action of the reaction wheel on the spacecraft
        
        return (np.zeros((N, 3)), spacecraft_torque)

    @staticmethod
    def differentiate(t: float, state: np.ndarray, state_indices: dict[str, tuple[int, int]], state_props: dict[str, np.ndarray]) -> np.ndarray:
        
        reaction_wheel_torque_cmd = np.clip(state[state_indices["REACTION_WHEEL_TORQUE_CMD"]], -state_props["REACTION_WHEEL_MAX_TORQUE"], state_props["REACTION_WHEEL_MAX_TORQUE"])
        reaction_wheel_dynamic_friction_coefficient = state_props["REACTION_WHEEL_DYNAMIC_FRICTION_COEFFICIENT"]
        reaction_wheel_angular_velocity = state[state_indices["REACTION_WHEEL_ANGULAR_VELOCITY"]]
        
        reaction_wheel_torque = reaction_wheel_torque_cmd - reaction_wheel_dynamic_friction_coefficient * reaction_wheel_angular_velocity
        
        reaction_wheel_inertia = state_props["REACTION_WHEEL_INERTIA"]
        d_reaction_wheel_angular_velocity = reaction_wheel_torque / reaction_wheel_inertia
        
        Y = state.shape[0]
        d_state = np.zeros(Y)
        
        d_state[state_indices["REACTION_WHEEL_ANGULAR_VELOCITY"]] = d_reaction_wheel_angular_velocity
        
        return d_state

class reactionWheel(ActuatorModel):
    """
    Reaction wheel actuator model.
    
    KEY -> REACTION_WHEEL
    
    state:
    - REACTION_WHEEL_ANGULAR_VELOCITY -> angular velocity of the reaction wheel in rad/s

    state_props:
    - REACTION_WHEEL_TORQUE_CMD -> command torque applied to the reaction wheel in Nm 
    - REACTION_WHEEL_ORIENTATION -> orientation of the reaction wheel in the body frame of the spacecraft
    - REACTION_WHEEL_INERTIA -> inertia of the reaction wheel in kg*m^2
    - REACTION_WHEEL_MAX_TORQUE -> maximum torque that can be applied to the reaction wheel in Nm
    - REACTION_WHEEL_MAX_VELOCITY -> maximum angular velocity of the reaction wheel in rad/s
    - REACTION_WHEEL_STATIC_FRICTION_COEFFICIENT -> static friction coefficient of the reaction wheel
    - REACTION_WHEEL_DYNAMIC_FRICTION_COEFFICIENT -> dynamic friction coefficient of the reaction wheel
    - REACTION_WHEEL_SPACECRAFT_MATRIX -> matrix of (0 | 1) that maps the reaction wheel to the corresponding spacecraft 
    """    
    
    def __init__(self, orientation: np.ndarray, intertia: float, max_torque: float, max_velocity: float, static_friction_coefficient: float, dynamic_friction_coefficient: float, initial_anglular_velocity: float = 0.0):
        super().__init__()
        
        self.orientation = orientation
        self.intertia = intertia 
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        self.static_friction_coefficient = static_friction_coefficient
        self.dynamic_friction_coefficient = dynamic_friction_coefficient
        
        self.initial_anglular_velocity = initial_anglular_velocity
        self.initial_torque_cmd = 0.0
        
        self.index: int = None # index of the reaction wheel in the global reaction wheels list (state vector)
        
    def initialize_count_manager(self, count: dict[str, int], manager: dict[str, Manager]):
        count["REACTION_WHEEL"] += 1
        manager["REACTION_WHEEL"] = reactionWheelManager  
        
    def initialize_state(self, state: np.ndarray, state_indices: dict[str, slice], state_props: dict[str, np.ndarray], indices: dict[str, int]):
        self.index = indices["REACTION_WHEEL"]
        indices["REACTION_WHEEL"] += 1
        
        reaction_wheel_angular_velocity = state[state_indices["REACTION_WHEEL_ANGULAR_VELOCITY"]].reshape(-1, 3)
        reaction_wheel_angular_velocity[self.index] = self.initial_anglular_velocity
        
        state_props["REACTION_WHEEL_TORQUE_CMD"][self.index] = self.initial_torque_cmd
        state_props["REACTION_WHEEL_ORIENTATION"][self.index] = self.orientation
        state_props["REACTION_WHEEL_INERTIA"][self.index] = self.intertia
        state_props["REACTION_WHEEL_MAX_TORQUE"][self.index] = self.max_torque
        state_props["REACTION_WHEEL_MAX_VELOCITY"][self.index] = self.max_velocity
        state_props["REACTION_WHEEL_STATIC_FRICTION_COEFFICIENT"][self.index] = self.static_friction_coefficient
        state_props["REACTION_WHEEL_DYNAMIC_FRICTION_COEFFICIENT"][self.index] = self.dynamic_friction_coefficient
        
        state_props["REACTION_WHEEL_SPACECRAFT_MATRIX"][self.spacecraft.index, self.index] = 1 
    
    def apply(self, command: dict[str, float]):
        if self.spacecraft is None:
            raise ValueError("Spacecraft is not set for the reaction wheel.")
        
        env = self.spacecraft.environment
        reaction_wheel_torque_cmd = env.state[env.state_indices["REACTION_WHEEL_TORQUE_CMD"]]
        reaction_wheel_torque_cmd[self.index] = command["REACTION_WHEEL_TORQUE_CMD"]

    