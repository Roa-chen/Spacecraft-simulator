import math

import numpy as np


class Camera:
    def __init__(self, position=None, yaw: float = 0.0, pitch: float = -30.0, move_speed: float = 500.0, mouse_sensitivity: float = 0.2):
        self.position = np.array(position if position is not None else [0.0, -1000.0, 500.0], dtype=float)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.mouse_sensitivity = mouse_sensitivity

    def get_flat_direction(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw)
        direction = np.array([
            - math.sin(yaw_rad),
            math.cos(yaw_rad),
            0.0,
        ], dtype=float)

        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0.0, 1.0, 0.0], dtype=float)

        return direction / norm

    def get_right(self) -> np.ndarray:
        forward = self.get_flat_direction()
        right = np.array([
            forward[1],
            -forward[0],
            0.0,
        ], dtype=float)

        norm = np.linalg.norm(right)
        if norm == 0:
            return np.array([1.0, 0.0, 0.0], dtype=float)

        return right / norm

    def update(self, dt: float, key_state: dict[str, bool], mouse_delta: tuple[float, float] | None = None):
        
        

        if mouse_delta is not None:
            delta_x, delta_y = mouse_delta
            self.yaw -= delta_x * self.mouse_sensitivity
            self.pitch -= delta_y * self.mouse_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            
            # print(f"yaw: {self.yaw: .1f} | pitch {self.pitch: .1f} | {self.position[0]: .1f} {self.position[1]: .1f} {self.position[2]: .1f} | {delta_x: .1f} {delta_y: .1f}")

        direction = np.zeros(3, dtype=float)
        forward = self.get_flat_direction()
        right = self.get_right()
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        if key_state.get("forward", False):
            direction += forward
        if key_state.get("backward", False):
            direction -= forward
        if key_state.get("left", False):
            direction -= right
        if key_state.get("right", False):
            direction += right
        if key_state.get("up", False):
            direction += up
        if key_state.get("down", False):
            direction -= up

        norm = np.linalg.norm(direction)
        if norm > 0:
            self.position += (direction / norm) * self.move_speed * dt

    def apply_to(self, camera_node) -> None:
        camera_node.setPos(*self.position)
        camera_node.setHpr(self.yaw, self.pitch, 0.0)