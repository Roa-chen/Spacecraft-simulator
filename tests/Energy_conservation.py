
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

if __name__ == "__main__":
    
    env = Earth_Spacecraft_elliptical()
    
    dt = 6
    T = 60 * 60 * 24
    
    t = []
    k = []
    p = []
    m = []
    d = []
    momentum = []
    
    dynamics = Dynamics()
    integrator = RK4(dynamics)
    simulation = Simulation(env, integrator, dt)
    
    earth = env.objects[0]
    spacecraft = env.objects[1]


    while env.get_time() < T:
        
        (k_en, p_en, m_en) = env.get_environment_energy()
        t.append(env.get_time())
        k.append(k_en)
        p.append(p_en)
        m.append(m_en)
        momentum.append(env.get_environment_momentum())
        
        dist = np.linalg.norm(spacecraft.state.position - earth.state.position)
        d.append(dist)

        simulation.step()

# fig, ax = plt.subplots(5)
        
# ax[0].plot(t, k); ax[0].set_title("Kinetic energy")
# ax[1].plot(t, p); ax[1].set_title("potential energy")
# ax[2].plot(t, m); ax[2].set_title("mechanical energy") 
# ax[3].plot(t, momentum); ax[3].set_title("momentum") 
# ax[4].plot(t, d); ax[4].set_title("distance") 

plt.plot(t, momentum)

plt.show()