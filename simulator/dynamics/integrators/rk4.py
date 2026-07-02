
from simulator.utils.quaternion import normalize_quaternions_state
from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator

class RK4(Integrator):
    def step(self, env: Environment, dt: float):
        
        k1 = self.dynamics.differentiate(env.get_time(), env.state, env.state_indices, env.state_props)
        k2 = self.dynamics.differentiate(env.get_time() + dt/2, env.state + k1 * dt/2, env.state_indices, env.state_props)
        k3 = self.dynamics.differentiate(env.get_time() + dt/2, env.state + k2 * dt/2, env.state_indices, env.state_props)
        k4 = self.dynamics.differentiate(env.get_time() + dt, env.state + k3 * dt, env.state_indices, env.state_props)

        env.state += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        
        normalize_quaternions_state(env.state, env.state_indices)
        
        
        

        