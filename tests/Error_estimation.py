
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

if __name__ == "__main__":
    
    T = 60 * 60 * 24
    
    d_em = []
    
    for dt in [6, 60]:
    
        # env = Earth_Spacecraft_circular()
        env = Earth_Spacecraft_elliptical()

        m = []
        
        dynamics = Dynamics()
        integrator = RK4(dynamics)
        simulation = Simulation(env, integrator, dt)


        while env.get_time() < T:
            
            (k_en, p_en, m_en) = env.get_environment_energy()
            m.append(m_en)
            simulation.step()

        d_em.append(m[-1] - m[0])
        
    print(d_em[1] / d_em[0])