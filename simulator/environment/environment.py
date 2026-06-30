
import numpy as np

from  simulator.core.clock import Clock
from  simulator.utils.constants import G
from  simulator.environment.object import Object
from  simulator.environment.spacecraft import Spacecraft

class Environment:
    def __init__(self):
        
        self.clock = Clock()
        self.objects: list[Object, Spacecraft] = []
        
        self.positions = np.zeros((0, 3))
        self.velocities = np.zeros((0, 3))
        self.attitudes = np.zeros((0, 4))
        self.angular_velocities = np.zeros((0, 3))
        
        self.masses = np.zeros(0)
        self.inertia_matrices = np.zeros((0, 3, 3))
        self.inertia_matrices_inv = np.zeros((0, 3, 3))

    def add_object(self, obj: Object):
        self.objects.append(obj)
        self.positions = np.concatenate((self.positions, obj.initial_state.position.reshape(1, 3)))
        self.velocities = np.concatenate((self.velocities, obj.initial_state.velocity.reshape(1, 3)))
        self.attitudes = np.concatenate((self.attitudes, obj.initial_state.attitude.reshape(1, 4)))
        self.angular_velocities = np.concatenate((self.angular_velocities, obj.initial_state.angular_velocity.reshape(1, 3)))

        self.masses = np.append(self.masses, obj.mass)
        
        if isinstance(obj, Spacecraft):
            self.inertia_matrices = np.concatenate((self.inertia_matrices, obj.inertia_matrix.reshape(1, 3, 3)), axis=0)
            self.inertia_matrices_inv = np.concatenate((self.inertia_matrices_inv, np.linalg.inv(obj.inertia_matrix).reshape(1, 3, 3)), axis=0)

        obj.environment = self
        obj.index = len(self.objects) - 1

    def add_objects(self, new_objects: list[object]):
        for obj in new_objects:
            self.add_object(obj)
        
    def get_time(self):
        return self.clock.get_time()
    
    def step_time(self, dt: float):
        self.clock.step(dt)
        
    def get_environment_momentum(self)->float:
        
        momentum = np.einsum('ni,n->ni', self.velocities, self.masses)
        return np.linalg.norm(momentum)
    
    def get_energies(self)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        N = len(self.objects)
        
        #kinetic energy
        kinetic_energies = 0.5 * self.masses * np.linalg.norm(self.velocities, axis=-1)**2
        
        # gravity potential energy
        
        # Calculate distances 
        pos1 = self.positions.reshape(N, 1, 3)
        pos2 = self.positions.reshape(1, N, 3)
        dist = pos2 - pos1
        dist = np.linalg.norm(dist, axis=2)
        
        # Handle division by 0
        dist += np.eye(N, dtype=float)*np.inf
        
        # Calculate G*m_i*m_j
        
        masses_t = np.transpose(self.masses)
        coefficients = - G * self.masses * masses_t
        
        # Unify
        
        potential_energies = np.sum(coefficients / dist, axis=1)
                    
        return kinetic_energies, potential_energies, kinetic_energies + potential_energies
    
    def get_environment_energy(self)->tuple[float, float, float]:
        
        kinetic_energies, potential_energies, _ = self.get_energies()
        
        kinetic_energy = np.sum(kinetic_energies)
        potential_energy = np.sum(potential_energies)
        mechanical_energy = kinetic_energy + potential_energy
                
        return kinetic_energy, potential_energy, mechanical_energy
            
        


