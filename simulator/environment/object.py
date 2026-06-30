import numpy as np

from simulator.core.state import State

class Object:
    def __init__(self, name: str, mass: float, radius: float, state: State) -> None:
        
        self.name = name
        self.mass = mass
        self.radius = radius
        self.state = state
        self.inertia_matrix = self.mass * np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ])