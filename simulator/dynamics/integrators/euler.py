
from simulator.utils.quaternion import normalize_quaternions_state
from simulator.environment.environment import Environment
from simulator.dynamics.integrators.integrator import Integrator
from simulator.dynamics.dynamics import Dynamics

class Euler(Integrator):
    @staticmethod
    def step(env: Environment, dt: float):
            
        d_state = Dynamics.differentiate(env.get_time(), env.state, env.state_indices, env.state_props)
        
        env.state += d_state * dt
        
        # Make sure quaternions are normalized
        normalize_quaternions_state(env.state, env.state_indices)

        