
from simulator.simulation.simulation import Simulation
from simulator.environment.environment import Environment
from simulator.environment.object import Object
from simulator.core.state import State
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.utils.constants import EARTH_MASS, EARTH_RADIUS, G
from simulator.visualization.renderer import Renderer

import numpy as np


def build_demo_environment():
    env = Environment()

    earth = Object(
        name="Earth",
        mass=EARTH_MASS,
        radius=EARTH_RADIUS,
        state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
    )

    orbital_altitude = 400_000.0
    satellite_position = np.array([EARTH_RADIUS + orbital_altitude, 0.0, 0.0], dtype=float)
    orbital_speed = np.sqrt(G * EARTH_MASS / np.linalg.norm(satellite_position))

    satellite = Object(
        name="Satellite",
        mass=1_000.0,
        radius=12_000.0,
        state=State(
            position=satellite_position,
            velocity=np.array([0.0, orbital_speed, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
    )

    env.add_objects([earth, satellite])
    return env


if __name__ == "__main__":

    env = build_demo_environment()
    
    dynamics = Dynamics()
    integrator = RK4(dynamics)
    simulation = Simulation(env, integrator, 1.0)


    renderer = Renderer(env, simulation)

    def simulation_task(task):
        simulation.step()
        return task.cont

    renderer.taskMgr.add(simulation_task, "simulation-task")
    renderer.run()





