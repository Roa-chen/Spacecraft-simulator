from simulator.core.state import State

class Object:
    def __init__(self, name: str, mass: float, radius: float, state: State) -> None:
        
        self.name = name
        self.mass = mass
        self.radius = radius
        self.state = state