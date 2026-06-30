import numpy as np

from simulator.core.state import State
from simulator.environment.object import Object
from simulator.environment.spacecraft import Spacecraft
from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator

class Euler(Integrator):
    def step(self, env: Environment, dt: float):
            
        props = {
            "MASS": env.masses,
            "INERTIA": env.inertia_matrices,
            "INERTIA_INV": env.inertia_matrices_inv,
        }
        
        # Computing next state

        (d_positions, d_velocities, d_attitude, d_angular_velocity) = self.dynamics.differentiate(env.get_time(), env.positions, env.velocities, env.attitudes, env.angular_velocities, props)
        
        env.positions += d_positions * dt
        env.velocities += d_velocities * dt
        env.attitudes += d_attitude * dt
        env.attitudes = np.einsum('ni,n->ni', env.attitudes, 1/np.linalg.norm(env.attitudes, axis=-1))
        env.angular_velocities += d_angular_velocity * dt
        

        