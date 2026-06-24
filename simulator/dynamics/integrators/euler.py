import numpy as np

from simulator.core.state import State
from simulator.environment.object import Object
from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator

class Euler(Integrator):
    def step(self, env: Environment, dt: float):
        
        objects = env.objects
        N = len(objects)
        
        # Packing data
        
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        masses = np.zeros(N)
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            positions[i] = state.position
            velocities[i] = state.velocity
            masses[i] = obj.mass
            
        props = {
            "masses": masses
        }
        
        # Computing next state

        (d_positions, d_velocities) = self.dynamics.differentiate(env.get_time(), positions, velocities, props)
        
        positions += d_positions * dt
        velocities += d_velocities * dt
        
        # Unpacking data
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            state.position = positions[i]
            state.velocity = velocities[i]
        

        