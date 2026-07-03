
from simulator.environment.environment import Environment
from simulator.dynamics.dynamics import Dynamics

"""
TODO: Angular momentum is not perfectly conserved. Thus it may be useful to integrate attitude several times per step to get more precision.
"""

class Integrator():
    @staticmethod
    def step(env: Environment, dt: float):
        raise NotImplementedError("Integrator step method must be implemented in subclasses.")