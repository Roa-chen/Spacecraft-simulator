
import numpy as np

from  simulator.core.clock import Clock
from  simulator.utils.constants import G
from  simulator.environment.object import Object

class Environment:
    def __init__(self):
        
        self.clock = Clock()
        self.objects: list[Object] = []

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_objects(self, new_objects: list[object]):
        self.objects.extend(new_objects)
        
    def get_time(self):
        return self.clock.get_time()
    
    def step_time(self, dt: float):
        self.clock.step(dt)
        
    def get_environment_momentum(self)->float:
        momentum = np.zeros(3)
        
        for obj in self.objects:
            momentum += obj.mass * obj.state.velocity
            
        return np.linalg.norm(momentum)
    
    def get_environment_energy(self)->tuple[float, float, float]:
        
        N = len(self.objects)
        
        kinetic_energy = 0
        potential_energy = 0
        
        for i, obj in enumerate(self.objects):
            #kinetic energy
            kinetic_energy += .5 * obj.mass * np.linalg.norm(obj.state.velocity)**2
            
            # gravity potential energy
            for j in range(i+1, N):
                other_obj = self.objects[j]
                d = np.linalg.norm(other_obj.state.position - obj.state.position)
                potential_energy += - G * obj.mass * other_obj.mass / d
                
        return kinetic_energy, potential_energy, kinetic_energy+potential_energy
        
    
    def get_energies(self)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        N = len(self.objects)
        
        kinetic_energy = np.zeros(N)
        potential_energy = np.zeros(N) # reference is 0 at infinity
        
        for i, obj in enumerate(self.objects):
            
            #kinetic energy
            kinetic_energy[i] = .5 * obj.mass * np.linalg.norm(obj.state.velocity)**2
            
            # gravity potential energy
            for other_obj in self.objects:
                if other_obj is not obj:
                    d = np.linalg.norm(other_obj.state.position - obj.state.position)
                    potential_energy[i] += - G * obj.mass * other_obj.mass / d
        
        mechanical_energy = kinetic_energy + potential_energy 
                    
        return kinetic_energy, potential_energy, mechanical_energy


