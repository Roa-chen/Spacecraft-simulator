
from  simulator.core.clock import Clock
from  simulator.environment.object import Object

class Environment:
    def __init__(self):
        
        self.clock = Clock()
        self.objects: list[Object] = []

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_objects(self, new_objects: list[object]):
        self.objects.extend(new_objects)
        
    def get_time(self):
        return self.clock.get_time()
    
    def step_time(self, dt: float):
        self.clock.step(dt)