
import numpy as np

from simulator.dynamics.forces.ForceModel import ForceModel
from simulator.utils.constants import G
from simulator.environment.environment import Environment
from simulator.environment.object import Object


class Gravity(ForceModel):
    def __init__(self):
        super().__init__()
        self.gravitational_constant = G

    def compute_force(self, obj: Object, env: Environment) -> np.ndarray:

        force = np.zeros(3)

        for other_obj in env.objects:
            if other_obj is not obj:
                r_vec = other_obj.state.position - obj.state.position
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 0:
                    force += (self.gravitational_constant * obj.mass * other_obj.mass / r_mag**3) * r_vec
        
        return force

        

        
        
        

        
        
        # force = np.zeros(3)
        # for other_obj in env.objects:
        #     if other_obj is not obj:
        #         r_vec = other_obj.position - obj.position
        #         r_mag = np.linalg.norm(r_vec)
        #         if r_mag > 0:
        #             force += (self.gravitational_constant * obj.mass * other_obj.mass / r_mag**2) * (r_vec / r_mag)
        # return force
