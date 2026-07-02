import numpy as np
from typing import TYPE_CHECKING

from simulator.core.state import State

if TYPE_CHECKING:
    from simulator.environment.environment import Environment

class Object:
    def __init__(self, name: str, mass: float, radius: float, initial_state: State) -> None:
        
        self.name = name    
        self.mass = mass
        self.radius = radius
        self.initial_state = initial_state
        self.inertia_matrix = self.mass * self.radius**2 * np.eye(3) / 2
        
        self.environment: Environment = None
        self.index: int = None
        
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