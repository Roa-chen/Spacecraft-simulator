import math
import numpy as np

import sys

from direct.gui.DirectGui import DirectButton
from direct.gui.DirectGui import DirectFrame
from direct.gui.DirectGui import OnscreenText
from direct.showbase.InputStateGlobal import inputState
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Geom
from panda3d.core import GeomNode
from panda3d.core import GeomTriangles
from panda3d.core import GeomVertexData
from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexWriter
from panda3d.core import NodePath
from panda3d.core import TextNode
from panda3d.core import Vec4
from panda3d.core import WindowProperties

from simulator.utils.constants import SCALE
from simulator.visualization.camera import Camera


class Renderer(ShowBase):

    def __init__(self, environment, simulation=None):
        super().__init__()

        self.env = environment
        self.simulation = simulation
        self.camera_controller = Camera()
        self._mouse_center = None
        self._sphere_template = self._create_sphere_template()
        self._control_panel = None
        self._status_text = None
        self.mouse_camera_mode = True

        self.disableMouse()
        self._configure_window()
        self._configure_lighting()
        self._bind_controls()
        self._create_ui()
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
        self._warp_mouse_to_center()

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
        inputState.watchWithModifiers("forward", "z")
        inputState.watchWithModifiers("backward", "s")
        inputState.watchWithModifiers("left", "q")
        inputState.watchWithModifiers("right", "d")
        inputState.watchWithModifiers("up", "space")
        inputState.watchWithModifiers("down", "shift")

        self.accept("f1", self.toggle_mouse_camera_mode)
        self.accept("f2", self.exit_window)

    def _warp_mouse_to_center(self):
        if self.win is None or self._mouse_center is None:
            return

        self.win.movePointer(0, self._mouse_center[0], self._mouse_center[1])

    def toggle_mouse_camera_mode(self):
        self.mouse_camera_mode = not self.mouse_camera_mode

        properties = WindowProperties()
        properties.setCursorHidden(self.mouse_camera_mode)
        if self.mouse_camera_mode:
            properties.setMouseMode(WindowProperties.M_confined)
            self._refresh_mouse_center()
            self._warp_mouse_to_center()
        else:
            properties.setMouseMode(WindowProperties.M_absolute)

        self.win.requestProperties(properties)
    
    def exit_window(self):
        sys.exit(0)

    def _create_ui(self):
        self._control_panel = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.08, 0.09, 0.12, 0.92),
            frameSize=(-0.02, 0.38, -1.0, 1.0),
            pos=(-1.48, 0.0, 0.0),
        )

        OnscreenText(
            parent=self._control_panel,
            text="Control Panel",
            pos=(0.02, 0.9),
            align=TextNode.ALeft,
            scale=0.055,
            fg=(1.0, 1.0, 1.0, 1.0),
        )

        DirectButton(
            parent=self._control_panel,
            text="Toggle mouse / camera (F1)",
            scale=0.04,
            pos=(0.19, 0.0, 0.78),
            frameSize=(-3.6, 3.6, -0.55, 0.55),
            command=self.toggle_mouse_camera_mode,
        )

        DirectButton(
            parent=self._control_panel,
            text="Pause / Resume",
            scale=0.045,
            pos=(0.19, 0.0, 0.60),
            frameSize=(-3.2, 3.2, -0.55, 0.55),
            command=self._on_pause_resume,
        )
        DirectButton(
            parent=self._control_panel,
            text="Reset view",
            scale=0.045,
            pos=(0.19, 0.0, 0.42),
            frameSize=(-3.2, 3.2, -0.55, 0.55),
            command=self._on_reset_view,
        )
        DirectButton(
            parent=self._control_panel,
            text="Reset simulation",
            scale=0.045,
            pos=(0.19, 0.0, 0.24),
            frameSize=(-3.2, 3.2, -0.55, 0.55),
            command=self._on_reset_simulation,
        )
        DirectButton(
            parent=self._control_panel,
            text="Follow target",
            scale=0.045,
            pos=(0.19, 0.0, 0.06),
            frameSize=(-3.2, 3.2, -0.55, 0.55),
            command=self._on_follow_target,
        )

        self._status_text = OnscreenText(
            parent=self._control_panel,
            text="",
            pos=(0.02, -0.08),
            align=TextNode.ALeft,
            scale=0.038,
            fg=(0.95, 0.96, 1.0, 1.0),
            mayChange=True,
        )

    def _create_sphere_template(self, slices: int = 72, stacks: int = 36) -> NodePath:
        format_ = GeomVertexFormat.getV3n3()
        vertex_data = GeomVertexData("sphere", format_, Geom.UHStatic)
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        normal_writer = GeomVertexWriter(vertex_data, "normal")

        for stack_index in range(stacks + 1):
            phi = math.pi * stack_index / stacks
            z = math.cos(phi)
            radius = math.sin(phi)

            for slice_index in range(slices + 1):
                theta = 2.0 * math.pi * slice_index / slices
                x = radius * math.cos(theta)
                y = radius * math.sin(theta)
                vertex_writer.addData3f(x, y, z)
                normal_writer.addData3f(x, y, z)

        triangles = GeomTriangles(Geom.UHStatic)

        def vertex_index(stack_index: int, slice_index: int) -> int:
            return stack_index * (slices + 1) + slice_index

        for stack_index in range(stacks):
            for slice_index in range(slices):
                top_left = vertex_index(stack_index, slice_index)
                bottom_left = vertex_index(stack_index + 1, slice_index)
                top_right = vertex_index(stack_index, slice_index + 1)
                bottom_right = vertex_index(stack_index + 1, slice_index + 1)

                triangles.addVertices(top_left, bottom_left, bottom_right)
                triangles.closePrimitive()
                triangles.addVertices(top_left, bottom_right, top_right)
                triangles.closePrimitive()

        geom = Geom(vertex_data)
        geom.addPrimitive(triangles)
        node = GeomNode("sphere-template")
        node.addGeom(geom)
        return NodePath(node)

    def _on_pause_resume(self):
        pass

    def _on_reset_view(self):
        self.camera_controller.position = np.array([0, 0, 500], dtype=float)
        self.camera_controller.yaw = 0.
        self.camera_controller.pitch = -89.

    def _on_reset_simulation(self):
        pass

    def _on_follow_target(self):
        pass

    def _get_mouse_delta(self):
        if self.win is None or not self.mouse_camera_mode:
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

    def _update_status_text(self):
        if self._status_text is None:
            return

        simulation_time = self.simulation.env.get_time() if self.simulation is not None else 0.0
        average_frame_rate = globalClock.getAverageFrameRate()
        camera_position = self.camera_controller.position

        self._status_text.setText(
            f"Simulation time: {simulation_time:,.1f} s\n"
            f"Frame rate: {average_frame_rate:,.1f} fps\n"
            f"Mouse mode: {'camera' if self.mouse_camera_mode else 'ui'}\n"
            f"Camera pos: {camera_position[0]:,.1f}, {camera_position[1]:,.1f}, {camera_position[2]:,.1f}\n"
            f"Camera yaw/pitch: {self.camera_controller.yaw:,.1f} / {self.camera_controller.pitch:,.1f}\n"
            f"Bodies: {len(self.env.objects)}"
        )

    def _frame_task(self, task):
        if self._mouse_center is None:
            self._refresh_mouse_center()

        dt = globalClock.getDt()
        mouse_delta = self._get_mouse_delta()
        self._update_camera_movement(dt, mouse_delta)
        self.camera_controller.apply_to(self.camera)
        self.update()
        self._update_status_text()

        return task.cont

    def _update_camera_movement(self, dt: float, mouse_delta):
        self.camera_controller.update(dt, self._movement_state(), mouse_delta)

    def _movement_state(self):
        return {
            "forward": inputState.isSet("forward"),
            "backward": inputState.isSet("backward"),
            "left": inputState.isSet("left"),
            "right": inputState.isSet("right"),
            "up": inputState.isSet("up"),
            "down": inputState.isSet("down"),
        }

    def update(self):
        for body in self.env.objects:
            pos = body.state.position * SCALE

            if not hasattr(body, "node"):
                body.node = self._sphere_template.copyTo(self.render)
                body.node.setLightOff()
                if body.name.lower() == "earth":
                    body.node.setColor(0.35, 0.65, 1.0, 1.0)
                else:
                    body.node.setColor(1.0, 0.95, 0.75, 1.0)

            body_scale = max(body.radius * SCALE, 1)
            # body_scale = body.radius * SCALE
            body.node.setScale(body_scale)
            body.node.setPos(*pos)
