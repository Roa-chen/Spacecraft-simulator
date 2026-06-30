import numpy as np

from simulator.core.state import State
from simulator.environment.object import Object
from simulator.environment.spacecraft import Spacecraft
from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator

class Euler(Integrator):
    def step(self, env: Environment, dt: float):
        
        objects = env.objects
        N = len(objects)
        
        # Packing data
        
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        attitude = np.zeros((N, 4))
        angular_velocity = np.zeros((N, 3))
        
        masses = np.zeros(N)
        I = np.zeros((N, 3, 3))
        I_inv = np.zeros((N, 3, 3))

        
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            positions[i] = state.position
            velocities[i] = state.velocity
            attitude[i] = state.attitude
            angular_velocity[i] = state.angular_velocity
            masses[i] = obj.mass
            
            if isinstance(obj, Spacecraft):
                I[i] = obj.inertia_matrix
                I_inv[i] = obj.inertia_matrix_inv
            
        props = {
            "MASS": masses,
            "INERTIA": I,
            "INERTIA_INV": I_inv
        }
        
        # Computing next state

        (d_positions, d_velocities, d_attitude, d_angular_velocity) = self.dynamics.differentiate(env.get_time(), positions, velocities, attitude, angular_velocity, props)
        
        positions += d_positions * dt
        velocities += d_velocities * dt
        attitude += d_attitude * dt
        attitude /= np.linalg.norm(attitude)        # normalizing quaternion
        angular_velocity += d_angular_velocity * dt
        
        # Unpacking data
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            state.position = positions[i]
            state.velocity = velocities[i]
            state.attitude = attitude[i]
            state.angular_velocity = angular_velocity[i]
        

        