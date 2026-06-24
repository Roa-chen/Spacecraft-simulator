
from simulator.dynamics.integrators.integrator import Integrator
from simulator.environment.environment import Environment
from simulator.dynamics.forces.gravity import Gravity

class Dynamics:
    def __init__(self, integrator: Integrator):
        self.integrator = integrator
        self.forces = [
            Gravity(),
        ]

    def step(self, env: Environment, dt: float):

        resultants = []


        # Compute forces

        for obj in env.objects:
            resultants.append(
                sum([
                    model.compute_force(obj, env) for model in self.forces
                ])
            )

        # Apply changes

        for obj, resultant in zip(env.objects, resultants):
            self.integrator.step(obj, resultant, dt)
