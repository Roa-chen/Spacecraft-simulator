
import numpy as np

from  simulator.core.clock import Clock
from  simulator.utils.constants import G
from  simulator.environment.object import Object
from  simulator.environment.spacecraft import Spacecraft

state_types = [
    "POSITION",
    "VELOCITY",
    "ATTITUDE",
    "ANGULAR_VELOCITY",
]

"""
TODO: improve data structure by placing spacecrafts contiguously in the state vector
"""

class Environment:
    def __init__(self):
        
        self.clock = Clock()
        self.objects: list[Object, Spacecraft] = []
        self.spacecrafts: list[Spacecraft] = []
        
        self.state_arrays = {
            "POSITION": np.zeros((0, 3)),
            "VELOCITY": np.zeros((0, 3)),
            "ATTITUDE": np.zeros((0, 4)),
            "ANGULAR_VELOCITY": np.zeros((0, 3))
        }
        
        self.state: np.ndarray = np.zeros(0)
        self.state_indices: dict[str, slice] = {}
        self.state_props: dict[str, np.ndarray] = {
            "MASS": np.zeros(0),
            "INERTIA_MATRIX": np.zeros((0, 3, 3)),
            "INERTIA_MATRIX_INV": np.zeros((0, 3, 3)),
            "SPACECRAFT_INDICES": np.zeros(0, dtype=int),
        }

    def add_object(self, obj: Object):
        self.objects.append(obj)
        obj.environment = self
        ind = len(self.objects) - 1
        obj.index = ind
        
        if isinstance(obj, Spacecraft):
            self.spacecrafts.append(obj)
            self.state_props["SPACECRAFT_INDICES"] = np.append(self.state_props["SPACECRAFT_INDICES"], ind)
            
        new_state_value_count = 0
            
        self.state_arrays["POSITION"] = np.concatenate((self.state_arrays["POSITION"], obj.initial_state.position.reshape(1, 3)))
        self.state_arrays["VELOCITY"] = np.concatenate((self.state_arrays["VELOCITY"], obj.initial_state.velocity.reshape(1, 3)))
        self.state_arrays["ATTITUDE"] = np.concatenate((self.state_arrays["ATTITUDE"], obj.initial_state.attitude.reshape(1, 4)))
        self.state_arrays["ANGULAR_VELOCITY"] = np.concatenate((self.state_arrays["ANGULAR_VELOCITY"], obj.initial_state.angular_velocity.reshape(1, 3)))

        self.state_props["MASS"] = np.append(self.state_props["MASS"], obj.mass)

        new_state_value_count += 3 + 3 + 4 + 3 + 1
        
        if isinstance(obj, Spacecraft):
            self.state_props["INERTIA_MATRIX"] = np.concatenate((self.state_props["INERTIA_MATRIX"], obj.inertia_matrix.reshape(1, 3, 3)), axis=0)
            self.state_props["INERTIA_MATRIX_INV"] = np.concatenate((self.state_props["INERTIA_MATRIX_INV"], np.linalg.inv(obj.inertia_matrix).reshape(1, 3, 3)), axis=0)

        self.state = np.zeros(self.state.size + new_state_value_count)
        
        i = 0
        for state_type in state_types:
            arr: np.ndarray = self.state_arrays[state_type]
            size = arr.size
            self.state[i: i + size] = arr.ravel()
            self.state_indices[state_type] = slice(i, i + size)
            i += size
        

    def add_objects(self, new_objects: list[object]):
        for obj in new_objects:
            self.add_object(obj)
        
    def get_time(self):
        return self.clock.get_time()
    
    def step_time(self, dt: float):
        self.clock.step(dt)
        
    # def get_environment_momentum(self)->float:
        
    #     momentum = np.einsum('ni,n->ni', self.velocities, self.masses)
    #     return np.linalg.norm(momentum)
    
    # def get_energies(self)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
    #     N = len(self.objects)
        
    #     #kinetic energy
    #     kinetic_energies = 0.5 * self.masses * np.linalg.norm(self.velocities, axis=-1)**2
        
    #     # gravity potential energy
        
    #     # Calculate distances 
    #     pos1 = self.positions.reshape(N, 1, 3)
    #     pos2 = self.positions.reshape(1, N, 3)
    #     dist = pos2 - pos1
    #     dist = np.linalg.norm(dist, axis=2)
        
    #     # Handle division by 0
    #     dist += np.eye(N, dtype=float)*np.inf
        
    #     # Calculate G*m_i*m_j
        
    #     masses_t = np.transpose(self.masses)
    #     coefficients = - G * self.masses * masses_t
        
    #     # Unify
        
    #     potential_energies = np.sum(coefficients / dist, axis=1)
                    
    #     return kinetic_energies, potential_energies, kinetic_energies + potential_energies
    
    # def get_environment_energy(self)->tuple[float, float, float]:
        
    #     kinetic_energies, potential_energies, _ = self.get_energies()
        
    #     kinetic_energy = np.sum(kinetic_energies)
    #     potential_energy = np.sum(potential_energies)
    #     mechanical_energy = kinetic_energy + potential_energy
                
    #     return kinetic_energy, potential_energy, mechanical_energy
            
        


