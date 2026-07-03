
import numpy as np

from simulator.core.state import State
from simulator.core.manager import Manager
from simulator.environment.object import Object, ObjectManager

class CelestialObjectManager(Manager):
    """
    KEY -> CELESTIAL_OBJECT
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

class CelestialObject(Object):
    """
    Represent all kind of celestial objects in the environment, including planets, moons, asteroids, etc.
    
    KEY -> CELESTIAL_OBJECT
    """
    
    def __init__(self, name: str, mass: float, radius: float, initial_state: State) -> None:
        super().__init__(name, mass, initial_state)
        
        self.radius = radius
        self.inertia_matrix = self.mass * self.radius**2 * np.eye(3) / 2
        
        self.celestial_object_index: int = None
        
    def initialize_count_manager(self, count: dict[str, int], manager: dict[str, Manager]):
        count["CELESTIAL_OBJECT"] += 1
        manager["CELESTIAL_OBJECT"] = CelestialObjectManager
        count["OBJECT"] += 1
        manager["OBJECT"] = ObjectManager
        
    def initialize_state(self, state: np.ndarray, state_indices: dict[str, slice], state_props: dict[str, np.ndarray], indices: dict[str, int]):
        self.index = indices["OBJECT"]
        indices["OBJECT"] += 1
        self.celestial_object_index = indices["CELESTIAL_OBJECT"]
        indices["CELESTIAL_OBJECT"] += 1
        
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