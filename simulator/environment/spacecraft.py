
import numpy as np

from simulator.actuators.actuatorModel import ActuatorModel
from simulator.core.state import State
from simulator.core.manager import Manager
from simulator.environment.object import Object, ObjectManager

class SpacecraftManager(Manager):
    """
    KEY -> SPACECRAFT
    """
    
    @staticmethod
    def get_state_required_properties(count: dict[str, int]) -> tuple[dict[str, int], dict[str, np.shape]]:
        N = count.get("OBJECT", 0)
        required_state = {
            "POSITION": 3 * N,
            "VELOCITY": 3 * N,
            "ATTITUDE": 4 * N,
            "ANGULAR_VELOCITY": 3 * N,
        }
        required_state_props = {
            "MASS": N,
            "INERTIA_MATRIX": (N, 3, 3),
            "INERTIA_MATRIX_INV": (N, 3, 3),
        }
        return required_state, required_state_props

class Spacecraft(Object):
    """
    Represent a spacecraft in the environment, it contains its own caracteristics such as mass, and actuators
    
    KEY -> SPACECRAFT    

    state:
    - POSITION -> position of the spacecraft in the world frame
    - VELOCITY -> velocity of the spacecraft in the world frame
    - ATTITUDE -> attitude of the spacecraft as a quaternion
    - ANGULAR_VELOCITY -> angular velocity of the spacecraft in the body frame

    state_props:
    - INERTIA_MATRIX -> inertia matrix of the spacecraft in the body frame
    - INERTIA_MATRIX_INV -> inverse of the inertia matrix of the spacecraft in the body frame
    """

    def __init__(self, name: str, mass: float, initial_state: State, actuators: list[ActuatorModel]) -> None:
        super().__init__(name, mass, initial_state)

        self.dimensions = np.array([10.0, 4.0, 3.0], dtype=float)
        
        a, b, c = self.dimensions / 2
        self.inertia_matrix = self.mass * np.array([
            [b**2 + c**2, 0, 0],
            [0, c**2 + a**2, 0],
            [0, 0, a**2 + b**2],
        ]) / 12
        self.inertia_matrix_inv = np.linalg.inv(self.inertia_matrix)
        
        self.actuators: list[ActuatorModel] = actuators
        
    def initialize_count_manager(self, count: dict[str, int], manager: dict[str, Manager]):
        count["SPACECRAFT"] += 1
        manager["SPACECRAFT"] = SpacecraftManager
        count["OBJECT"] += 1
        manager["OBJECT"] = ObjectManager

        for actuator in self.actuators:
            actuator.initialize_count_manager(count, manager)
        
    def initialize_state(self, state: np.ndarray, state_indices: dict[str, slice], state_props: dict[str, np.ndarray], indices: dict[str, int]):
        self.index = indices["OBJECT"]
        indices["OBJECT"] += 1
        self.spacecraft_index = indices["SPACECRAFT"]
        indices["SPACECRAFT"] += 1
        
        position = state[state_indices["POSITION"]].reshape(-1, 3)
        velocity = state[state_indices["VELOCITY"]].reshape(-1, 3)
        attitude = state[state_indices["ATTITUDE"]].reshape(-1, 4)
        angular_velocity = state[state_indices["ANGULAR_VELOCITY"]].reshape(-1, 3)
        
        position[self.index] = self.initial_state.position
        velocity[self.index] = self.initial_state.velocity
        attitude[self.index] = self.initial_state.attitude
        angular_velocity[self.index] = self.initial_state.angular_velocity
        
        state_props["MASS"][self.index] = self.mass
        state_props["INERTIA_MATRIX"][self.index] = self.inertia_matrix
        state_props["INERTIA_MATRIX_INV"][self.index] = np.linalg.inv(self.inertia_matrix)
        
        for actuator in self.actuators:
            actuator.initialize_state(state, state_indices, state_props, indices)
    
    def get_position(self) -> np.ndarray:
        position = self.environment.state[self.environment.state_indices["POSITION"]].reshape(-1, 3)
        return position[self.index]
    
    def get_velocity(self) -> np.ndarray:
        velocity = self.environment.state[self.environment.state_indices["VELOCITY"]].reshape(-1, 3)
        return velocity[self.index]

    def get_attitude(self) -> np.ndarray:
        attitude = self.environment.state[self.environment.state_indices["ATTITUDE"]].reshape(-1, 4)
        return attitude[self.index]

    def get_angular_velocity(self) -> np.ndarray:
        angular_velocity = self.environment.state[self.environment.state_indices["ANGULAR_VELOCITY"]].reshape(-1, 3)
        return angular_velocity[self.index]
    
    def add_actuator(self, actuator: ActuatorModel):
        self.actuators.append(actuator)

    def add_actuators(self, new_actuators: list[ActuatorModel]):
        for actuator in new_actuators:
            self.add_actuator(actuator)