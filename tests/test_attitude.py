
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

if __name__ == "__main__":
    
    T = 60
    
    t = []
    attitude = []
    angular_velocity = []
    
    
    env = Earth_Spacecraft_circular()
    # env = Earth_Spacecraft_elliptical()
    
    dt = 1
    m = []
    
    dynamics = Dynamics()
    integrator = RK4(dynamics)
    simulation = Simulation(env, integrator, dt)


    while env.get_time() < T:
        
        t.append(env.get_time())
        attitude.append(env.objects[1].get_attitude())
        angular_velocity.append(env.objects[1].get_angular_velocity())
        simulation.step()
        
    plt.plot(t, attitude)
    plt.show()

