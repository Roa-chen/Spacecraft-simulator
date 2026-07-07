
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer

import numpy as np

"""
TODO: Reaction wheel are buggy
TODO: Find realistic values for reaction wheel parameters
"""


if __name__ == "__main__":

    env = Spacecraft_Only()
    
    spacecaft = env.objects[0]
    reacWh = spacecaft.actuators[0]
    
    reacWh.apply({"REACTION_WHEEL_TORQUE_CMD": 0.01})
    
    dt = 1.0
    integrator = Euler
    simulation = Simulation(env, integrator, dt)
    
    # N = 10_000
    # T = 0
    
    # t = []
    # s_ang_vel = []
    # w_ang_vel = []
    
    # for _ in range(N):
        
    #     t.append(T)    
    #     s_ang_vel.append(np.linalg.norm(spacecaft.get_angular_velocity()))
    #     w_ang_vel.append(np.linalg.norm(reacWh.get_angular_velocity()))
        
    #     simulation.step()
    #     T += dt

    # fig, ax = plt.subplots(2)

    # ax[0].plot(t, s_ang_vel); ax[0].set_title("Spacecraft")
    # ax[1].plot(t, w_ang_vel); ax[1].set_title("ReactionWheel")

    # plt.show()
    
    renderer = Renderer(env, simulation)
    renderer.camera_controller.position = np.array([0.0, -7, 4], dtype=float)
    renderer.camera_controller.move_speed = 20.
    renderer.run()
