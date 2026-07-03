
from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

import numpy as np



if __name__ == "__main__":

    env = Earth_Spacecraft_circular()
    # env = Earth_Spacecraft_elliptical()
    env = Earth_Two_Spacecrafts_circular()
    
    dynamics = Dynamics()
    integrator = RK4(dynamics)
    simulation = Simulation(env, integrator, 6.0)


    renderer = Renderer(env, simulation)

    renderer.run()





