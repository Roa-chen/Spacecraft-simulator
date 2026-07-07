import numpy as np

from simulator.actuators.reactionWheel import ReactionWheel
from simulator.utils.constants import *
from simulator.environment.environment import Environment
from simulator.environment.celestialObject import CelestialObject
from simulator.environment.spacecraft import Spacecraft
from simulator.core.state import State


def Earth_Spacecraft_circular(orbital_altitude: float = 400_000.0):
    env = Environment()

    earth = CelestialObject(
        name="Earth",
        mass=EARTH_MASS,
        radius=EARTH_RADIUS,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
    )

    satellite_position = np.array(
        [EARTH_RADIUS + orbital_altitude, 0.0, 0.0], dtype=float
    )
    orbital_speed = np.sqrt(G * EARTH_MASS / np.linalg.norm(satellite_position))

    satellite = Spacecraft(
        name="Satellite",
        mass=1_000.0,
        initial_state=State(
            position=satellite_position,
            velocity=np.array([0.0, orbital_speed, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([5e-3, 0.0, 0.0], dtype=float),
        ),
        actuators=[],
    )

    env.add_objects([earth, satellite])

    env.initialize_state()

    return env


def Earth_Spacecraft_elliptical(
    min_orbital_radius: float = 8e6, max_orbital_radius: float = 30e6
):
    env = Environment()

    earth = CelestialObject(
        name="Earth",
        mass=EARTH_MASS,
        radius=EARTH_RADIUS,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
    )

    satellite_position = np.array([max_orbital_radius, 0.0, 0.0], dtype=float)
    orbital_speed = np.sqrt(
        2
        * G
        * EARTH_MASS
        * min_orbital_radius
        / (max_orbital_radius * (max_orbital_radius + min_orbital_radius))
    )

    satellite = Spacecraft(
        name="Satellite",
        mass=1_000.0,
        initial_state=State(
            position=satellite_position,
            velocity=np.array([0.0, orbital_speed, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
        actuators=[],
    )

    env.add_objects([earth, satellite])

    env.initialize_state()

    return env


def Earth_Two_Spacecrafts_circular(orbital_altitude: float = 400_000.0):
    env = Environment()

    earth = CelestialObject(
        name="Earth",
        mass=EARTH_MASS,
        radius=EARTH_RADIUS,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
    )

    satellite_position = EARTH_RADIUS + orbital_altitude
    orbital_speed = np.sqrt(G * EARTH_MASS / np.linalg.norm(satellite_position))

    satellite_1 = Spacecraft(
        name="Satellite 1",
        mass=1_000.0,
        initial_state=State(
            position=np.array([satellite_position, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, orbital_speed, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([5e-3, 0.0, 0.0], dtype=float),
        ),
        actuators=[],
    )

    satellite_2 = Spacecraft(
        name="Satellite 2",
        mass=1_000.0,
        initial_state=State(
            position=np.array([0.0, -satellite_position, 0.0], dtype=float),
            velocity=np.array([orbital_speed, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 5e-3], dtype=float),
        ),
        actuators=[],
    )

    env.add_objects([earth, satellite_1, satellite_2])

    env.initialize_state()

    return env

def Spacecraft_Only():
    env = Environment()
    
    reaction_wheel = ReactionWheel(
        orientation=np.array([0.0, 0.0, 10.0], dtype=float),
        intertia=1.0,
        max_torque=10000.0,
        max_velocity=10000.0,
        static_friction_coefficient=0.0,
        dynamic_friction_coefficient=0.0,
        initial_anglular_velocity=0.0,
        initial_torque_cmd=-0.00
    )

    satellite = Spacecraft(
        name="Satellite",
        mass=1_000.0,
        initial_state=State(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity=np.array([0.0, 0.0, 0.0], dtype=float),
            attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_velocity=np.array([0.0, 0.0, 0.0], dtype=float),
        ),
        actuators=[reaction_wheel],
    )

    env.add_objects([satellite])

    env.initialize_state()
    
    print("Initial command:", env.state_props["REACTION_WHEEL_TORQUE_CMD"][reaction_wheel.index])

    return env
