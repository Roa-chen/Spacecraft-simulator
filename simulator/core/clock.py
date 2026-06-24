
class Clock:

    def __init__(self):
        
        self.time: float = 0

    def step(self, dt: float):
        self.time += dt

    def get_time(self) -> float:
        return self.time