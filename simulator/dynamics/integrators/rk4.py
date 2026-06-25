import numpy as np

from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator

"""
TODO: Change the data structure of the environment to save directly the global state vector to get rid of the pack/unpack process.
"""

class RK4(Integrator):
    def step(self, env: Environment, dt: float):
        
        objects = env.objects
        N = len(objects)
        
        # Packing data
        
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        masses = np.zeros(N)
        
        for i, obj in enumerate(env.objects):
            state = obj.state
            
            positions[i] = state.position
            velocities[i] = state.velocity
            masses[i] = obj.mass
            
        props = {
            "masses": masses
        }
        
        # Computing next state

        (k1_pos, k1_vel) = self.dynamics.differentiate(env.get_time(), positions, velocities, props)
        (k2_pos, k2_vel) = self.dynamics.differentiate(env.get_time() + dt/2, positions + k1_pos * dt/2, velocities + k1_vel * dt/2, props)
        (k3_pos, k3_vel) = self.dynamics.differentiate(env.get_time() + dt/2, positions + k2_pos * dt/2, velocities + k2_vel * dt/2, props)
        (k4_pos, k4_vel) = self.dynamics.differentiate(env.get_time() + dt, positions + k3_pos * dt, velocities + k3_vel * dt, props)
        
        
        positions += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) * dt / 6
        velocities += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) * dt / 6
        
        # Unpacking data
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            state.position = positions[i]
            state.velocity = velocities[i]
        

        