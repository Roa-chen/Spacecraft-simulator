import numpy as np

from simulator.environment.environment import Environment
from simulator.environment.spacecraft import Spacecraft
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
        attitude = np.zeros((N, 4))
        angular_velocity = np.zeros((N, 3))
        
        masses = np.zeros(N)
        I = np.zeros((N, 3, 3))
        I_inv = np.zeros((N, 3, 3))
        
        for i, obj in enumerate(env.objects):
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

        (k1_pos, k1_vel, k1_att, k1_ang) = self.dynamics.differentiate(env.get_time(), positions, velocities, attitude, angular_velocity, props)
        (k2_pos, k2_vel, k2_att, k2_ang) = self.dynamics.differentiate(env.get_time() + dt/2, positions + k1_pos * dt/2, velocities + k1_vel * dt/2, attitude + k1_att * dt/2, angular_velocity + k1_ang * dt/2, props)
        (k3_pos, k3_vel, k3_att, k3_ang) = self.dynamics.differentiate(env.get_time() + dt/2, positions + k2_pos * dt/2, velocities + k2_vel * dt/2, attitude + k2_att * dt/2, angular_velocity + k2_ang * dt/2, props)
        (k4_pos, k4_vel, k4_att, k4_ang) = self.dynamics.differentiate(env.get_time() + dt, positions + k3_pos * dt, velocities + k3_vel * dt, attitude + k3_att * dt/2, angular_velocity + k3_ang * dt/2, props)
        
        
        positions += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) * dt / 6
        velocities += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) * dt / 6
        attitude += (k1_att + 2 * k2_att + 2 * k3_att + k4_att) * dt / 6
        attitude = np.einsum('ni,n->ni', attitude, 1/np.linalg.norm(attitude, axis=-1))
        angular_velocity += (k1_ang + 2 * k2_ang + 2 * k3_ang + k4_ang) * dt / 6
        'ni,n->ni'
        
        # Unpacking data
        
        for i in range(N):
            obj = objects[i]
            state = obj.state
            
            state.position = positions[i]
            state.velocity = velocities[i]
            state.attitude = attitude[i]
            state.angular_velocity = angular_velocity[i]
        

        