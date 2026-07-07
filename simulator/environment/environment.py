
import numpy as np
from collections import defaultdict 

from simulator.utils.constants import G
from simulator.core.clock import Clock
from simulator.core.manager import Manager
from simulator.dynamics.actions.actionModel import ActionModel
from simulator.actuators.actuatorModel import ActuatorModelManager
from simulator.environment.celestialObject import CelestialObject
from simulator.environment.spacecraft import Spacecraft


class Environment:
    def __init__(self):
        
        self.clock = Clock()
        
        self.count = defaultdict(int)
        self.manager: dict[str, Manager] = dict()
        
        self.objects: list[CelestialObject | Spacecraft] = []
        
        self.state: np.ndarray = None
        self.state_indices: dict[str, slice] = {}
        self.state_props: dict[str, np.ndarray] = {}
        
    def add_object(self, obj: CelestialObject):
        self.objects.append(obj)
        obj.environment = self
        obj.initialize_count_manager(self.count, self.manager)

    def add_objects(self, new_objects: list[CelestialObject]):
        for obj in new_objects:
            self.add_object(obj)

    def initialize_state(self):
        
        required_state = {}
        required_state_props = {}
        
        for manager in self.manager.values():
            required = manager.get_state_required_properties(self.count)
            required_state.update(required[0])
            required_state_props.update(required[1])
            
        Y = 0
        for name, size in required_state.items():
            self.state_indices[name] = slice(Y, Y + size)
            Y += size
        self.state = np.zeros(Y)
            
        for name, shape in required_state_props.items():
            self.state_props[name] = np.zeros(shape)
            
        self.objects.sort(key=lambda obj: isinstance(obj, Spacecraft), reverse=True)
        
        indices = defaultdict(int)
        
        for obj in self.objects:
            obj.initialize_state(self.state, self.state_indices, self.state_props, indices)
        

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
            
        


