import numpy as np

from simulator.environment.environment import Environment
from simulator.environment.spacecraft import Spacecraft
from simulator.dynamics.integrators.integrator import Integrator

class RK4(Integrator):
    def step(self, env: Environment, dt: float):
        
        props = {
            "MASS": env.masses,
            "INERTIA": env.inertia_matrices,
            "INERTIA_INV": env.inertia_matrices_inv
        }
        
        # Computing next state

        (k1_pos, k1_vel, k1_att, k1_ang) = self.dynamics.differentiate(env.get_time(), env.positions, env.velocities, env.attitudes, env.angular_velocities, props)
        (k2_pos, k2_vel, k2_att, k2_ang) = self.dynamics.differentiate(env.get_time() + dt/2, env.positions + k1_pos * dt/2, env.velocities + k1_vel * dt/2, env.attitudes + k1_att * dt/2, env.angular_velocities + k1_ang * dt/2, props)
        (k3_pos, k3_vel, k3_att, k3_ang) = self.dynamics.differentiate(env.get_time() + dt/2, env.positions + k2_pos * dt/2, env.velocities + k2_vel * dt/2, env.attitudes + k2_att * dt/2, env.angular_velocities + k2_ang * dt/2, props)
        (k4_pos, k4_vel, k4_att, k4_ang) = self.dynamics.differentiate(env.get_time() + dt, env.positions + k3_pos * dt, env.velocities + k3_vel * dt, env.attitudes + k3_att * dt/2, env.angular_velocities + k3_ang * dt/2, props)

        
        env.positions += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) * dt / 6
        env.velocities += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) * dt / 6
        env.attitudes += (k1_att + 2 * k2_att + 2 * k3_att + k4_att) * dt / 6
        env.attitudes = np.einsum('ni,n->ni', env.attitudes, 1/np.linalg.norm(env.attitudes, axis=-1))
        env.angular_velocities += (k1_ang + 2 * k2_ang + 2 * k3_ang + k4_ang) * dt / 6
        
        

        