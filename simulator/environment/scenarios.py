
import numpy as np

from simulator.utils.constants import *
from simulator.environment.environment import Environment
from simulator.environment.object import Object
from simulator.core.state import State

def Earth_Spacecraft_circular(orbital_altitude: float = 400_000.0):
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

def Earth_Spacecraft_elliptical(min_orbital_radius: float = 8e6, max_orbital_radius: float = 30e6):
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

    satellite_position = np.array([max_orbital_radius, 0.0, 0.0], dtype=float)
    orbital_speed = np.sqrt(2* G * EARTH_MASS * min_orbital_radius / (max_orbital_radius * (max_orbital_radius + min_orbital_radius)))

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
