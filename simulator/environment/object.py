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
        return self.environment.positions[self.index]
    
    def get_velocity(self) -> np.ndarray:
        return self.environment.velocities[self.index]
    
    def get_attitude(self) -> np.ndarray:
        return self.environment.attitudes[self.index]
    
    def get_angular_velocity(self) -> np.ndarray:
        return self.environment.angular_velocities[self.index]