import numpy as np
from typing import TYPE_CHECKING

from simulator.core.state import State

if TYPE_CHECKING:
    from simulator.environment.environment import Environment

class Spacecraft:
    def __init__(self, name: str, mass: float, initial_state: State) -> None:
        
        self.name = name
        self.mass = mass
        self.initial_state = initial_state
        
        """
        Dimensions:
        10m * 4m * 3m
        """
        
        a = 10
        b = 4
        c = 3
        
        self.inertia_matrix = self.mass * np.array([
            [b**2 + c**2, 0, 0],
            [0, c**2 + a**2, 0],
            [0, 0, a**2 + b**2],
        ]) / 12
        
        self.inertia_matrix_inv = np.linalg.inv(self.inertia_matrix)
        
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