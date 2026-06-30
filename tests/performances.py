
import time
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

if __name__ == "__main__":
    
    N = 100
    
    for dt in [1, 10, 100]:
    
        # env = Earth_Spacecraft_circular()
        env = Earth_Spacecraft_elliptical()

        dynamics = Dynamics()
        integrator = RK4(dynamics)
        simulation = Simulation(env, integrator, dt)

        t0 = time.time()
        
        for _ in range(N):
            simulation.step()

        t1 = time.time()
        
        print(f"dt = {dt}, time taken for {N} steps: {t1 - t0:.4f} seconds, time per step: {(t1 - t0) / N:.6f} seconds")
        
        
"""
dt = 1, time taken for 10000 steps: 4.4701 seconds, time per step: 0.000447 seconds
dt = 10, time taken for 10000 steps: 4.9463 seconds, time per step: 0.000495 seconds
dt = 100, time taken for 10000 steps: 4.9937 seconds, time per step: 0.000499 second

If fps=60, then dt_sim = 1/60 = 0.0166667 s. This means that the simulation can run at a speed of 37 steps per second.
"""