
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec4
from panda3d.core import WindowProperties

from simulator.utils.constants import SCALE
from simulator.visualization.camera import Camera


class Renderer(ShowBase):

    def __init__(self, environment):
        super().__init__()

        self.env = environment
        self.camera_controller = Camera()
        self.key_state = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }
        self._mouse_center = None

        self.disableMouse()
        self._configure_window()
        self._configure_lighting()
        self._bind_controls()
        self.camLens.setFov(70)
        self.camLens.setNearFar(1.0, 1_000_000.0)
        self.camera_controller.apply_to(self.camera)

        self.taskMgr.add(self._frame_task, "frame-task")

    def _configure_window(self):
        properties = WindowProperties()
        properties.setCursorHidden(True)
        properties.setMouseMode(WindowProperties.M_confined)
        self.win.requestProperties(properties)
        self._refresh_mouse_center()
        self.win.movePointer(0, self._mouse_center[0], self._mouse_center[1])

    def _refresh_mouse_center(self):
        if self.win is None:
            self._mouse_center = None
            return

        properties = self.win.getProperties()
        self._mouse_center = (properties.getXSize() // 2, properties.getYSize() // 2)

    def _configure_lighting(self):
        ambient = AmbientLight("ambient-light")
        ambient.setColor(Vec4(0.35, 0.35, 0.4, 1.0))
        ambient_node = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_node)

        directional = DirectionalLight("sun-light")
        directional.setColor(Vec4(0.9, 0.9, 0.85, 1.0))
        directional_node = self.render.attachNewNode(directional)
        directional_node.setHpr(45.0, -45.0, 0.0)
        self.render.setLight(directional_node)

        self.setBackgroundColor(0.02, 0.03, 0.07)

    def _bind_controls(self):
        bindings = {
            "z": ("forward", True),
            "z-up": ("forward", False),
            "s": ("backward", True),
            "s-up": ("backward", False),
            "q": ("left", True),
            "q-up": ("left", False),
            "d": ("right", True),
            "d-up": ("right", False),
            "space": ("up", True),
            "space-up": ("up", False),
            "shift": ("down", True),
            "shift-up": ("down", False),
        }

        for event_name, (key_name, pressed) in bindings.items():
            self.accept(event_name, self._set_key, [key_name, pressed])

    def _set_key(self, key_name: str, pressed: bool):
        self.key_state[key_name] = pressed

    def _get_mouse_delta(self):
        if self.win is None:
            return None

        if self._mouse_center is None:
            self._refresh_mouse_center()

        if self._mouse_center is None:
            return None

        pointer = self.win.getPointer(0)
        delta_x = pointer.getX() - self._mouse_center[0]
        delta_y = pointer.getY() - self._mouse_center[1]

        if delta_x != 0 or delta_y != 0:
            self.win.movePointer(0, self._mouse_center[0], self._mouse_center[1])

        return delta_x, delta_y

    def _frame_task(self, task):
        if self._mouse_center is None:
            self._refresh_mouse_center()

        dt = globalClock.getDt()
        mouse_delta = self._get_mouse_delta()
        self.camera_controller.update(dt, self.key_state, mouse_delta)
        self.camera_controller.apply_to(self.camera)
        self.update()

        return task.cont

    def update(self):

        for body in self.env.objects:
            pos = body.state.position * SCALE

            if not hasattr(body, "node"):
                body.node = self.loader.loadModel("models/misc/sphere")
                body.node.reparentTo(self.render)
                body.node.setLightOff()
                if body.name.lower() == "earth":
                    body.node.setColor(0.35, 0.65, 1.0, 1.0)
                else:
                    body.node.setColor(1.0, 0.95, 0.75, 1.0)

            body_scale = max(body.radius * SCALE, 1)
            # body_scale = max(body.radius * SCALE * 4.0, 4.0)
            body.node.setScale(body_scale)
            body.node.setPos(*pos)