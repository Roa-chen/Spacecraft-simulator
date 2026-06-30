
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer
    

if __name__ == "__main__":
    
    T = 20_000
    
    t = []
    attitude = []
    angular_velocity = []
    
    env = Environment()

    satellite = Spacecraft(
        name="Satellite",
        mass=1_000.0,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, np.pi*1e-3, ], dtype=float),
        ),
    )

    env.add_object(satellite)
    
    dt = 10
    dynamics = Dynamics()
    integrator = Euler(dynamics)
    simulation = Simulation(env, integrator, dt)

    # while env.get_time() < T:
        
    #     t.append(env.get_time())
    #     attitude.append(satellite.get_attitude())
    #     angular_velocity.append(satellite.get_angular_velocity())
    #     simulation.step()
        
    # plt.plot(t, attitude)
    # plt.show()
    
    renderer = Renderer(env, simulation)
    renderer.sim_time_per_sec = 1000.0
    renderer.run()
