import numpy as np
from typing import TYPE_CHECKING

from simulator.core.state import State
from simulator.core.manager import Manager

if TYPE_CHECKING:
    from simulator.environment.environment import Environment

class ObjectManager(Manager):
    """
    KEY -> OBJECT
    """
    @staticmethod
    def get_state_required_properties(count: dict[str, int]) -> tuple[dict[str, int], dict[str, np.shape]]:
        return {}, {}
    
class Object:
    """
    Represent all kind of physical objects in the environment.
    
    KEY -> OBJECT
    """
    
    def __init__(self, name: str, mass: float, initial_state: State) -> None:
        
        self.name = name    
        self.mass = mass
        self.initial_state = initial_state
        
        self.inertia_matrix = np.zeros((3, 3))
        
        self.environment: Environment = None
        self.index: int = None
        
    def initialize_count_manager(self, count: dict[str, int], manager: dict[str, Manager]):
        raise NotImplementedError("This method should be overridden by subclasses.")
        