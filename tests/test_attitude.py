
import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.scenarios import *
from simulator.simulation.simulation import Simulation
from simulator.dynamics.dynamics import Dynamics
from simulator.dynamics.integrators.euler import Euler
from simulator.dynamics.integrators.rk4 import RK4
from simulator.visualization.renderer import Renderer
    
    
"""
Conclusion:
Unstability if the spacecraft spins around its intermediate principal axis. The simulation is stable around the other two axes.
Considering a working spacecraft controller, the spacecraft shouldn't be spinning too fast, thus the results are acceptable.
"""

if __name__ == "__main__":
    
    # T = 200
    # T = 70_000
    T = 460_000
    # T = 470_000
    # T = 2_000_000
    n = 0
    
    t = []
    attitude = []
    angular_velocity = []
    angular_momentum = []
    
    env = Environment()

    satellite = Spacecraft(
        name="Satellite",
        mass=1_000.0,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            # angular_velocity=np.array([np.pi*1e-3 / 1, np.pi*1e-3 / 2, np.pi*1e-3 / 3], dtype=float),
            angular_velocity=np.array([np.pi*1e-3 / 2, 0, np.pi*1e-3 / 1], dtype=float),
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
    #     angular_momentum.append(np.linalg.norm(np.matmul(satellite.inertia_matrix, satellite.get_angular_velocity()), axis=-1))
    #     simulation.step()
        
    #     n+=1
    #     if n % 2000 == 0:
    #         print(f"Time: {env.get_time():.2f} s, Step: {n}, Attitude: {satellite.get_attitude()}, Angular Velocity: {satellite.get_angular_velocity()}")
    #         Iw = np.einsum('ij,j->i', satellite.inertia_matrix, satellite.get_angular_velocity())
    #         cross = np.cross(satellite.get_angular_velocity(), Iw, axis=-1)
    #         print(f"Norm of attitude quaternion: {np.linalg.norm(satellite.get_attitude())}")
    #         print(f"Iw: {Iw}, Cross: {cross}\n")
        
    # plt.plot(t, angular_momentum)
    # plt.show()
    
    renderer = Renderer(env, simulation)
    renderer.camera_controller.position = np.array([0., -10., 5.])
    renderer.camera_controller.move_speed = 20.
    renderer.sim_time_per_sec = 1000.0
    renderer.run()
