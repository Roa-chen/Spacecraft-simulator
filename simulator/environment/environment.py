
from  simulator.core.state import State
from  simulator.environment.object import Object

class Environment:
    def __init__(self):
        
        self.objects: list[Object] = []

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_objects(self, objetcs: list[object]):
        self.objects.extend(self.objects)