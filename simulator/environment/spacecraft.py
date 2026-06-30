import numpy as np

from simulator.core.state import State

class Spacecraft:
    def __init__(self, name: str, mass: float, state: State) -> None:
        
        self.name = name
        self.mass = mass
        self.state = state
        
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