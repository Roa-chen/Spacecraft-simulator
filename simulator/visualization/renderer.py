import math
import numpy as np

import sys

from direct.gui.DirectGui import DirectButton
from direct.gui.DirectGui import DirectEntry
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
from panda3d.core import Quat
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import WindowProperties

from simulator.utils.constants import SCALE
from simulator.environment.spacecraft import Spacecraft
from simulator.visualization.camera import Camera


class Renderer(ShowBase):

    def __init__(self, environment, simulation=None):
        super().__init__()

        self.env = environment
        self.simulation = simulation
        self.camera_controller = Camera()
        self._mouse_center = None
        self._sphere_template = self._create_sphere_template()
        self._brick_template = self._create_brick_template()
        self._control_panel = None
        self._status_text = None
        self.mouse_camera_mode = True
        
        self.sim_time_per_sec = 1000.0
        self.pause = False

        self.disableMouse()
        self._configure_window()
        self._configure_lighting()
        self._bind_controls()
        self._create_ui()
        self.camLens.setFov(70)
        self.camLens.setNearFar(1.0, 1_000_000.0)
        self.camera_controller.apply_to(self.camera)

        self.taskMgr.add(self._frame_task, "frame-task")
        self.taskMgr.add(self.simulation_task, "simulation-task")

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
        # 1. State for toggling the UI
        self.ui_is_open = True
        
        # 2. Base frame anchored to the true Top-Left of the window
        self._control_panel = DirectFrame(
            parent=self.a2dTopLeft,
            frameColor=(0.08, 0.09, 0.12, 0.95),
            frameSize=(0, 0.75, -2.0, 0),  # 0.75 wide, extends to bottom of screen
            pos=(0, 0, 0),
        )

        # 3. Auto-layout state
        self._ui_current_z = -0.15      # Starting vertical position
        self._ui_margin_x = 0.05        # Left indent for text
        self._ui_center_x = 0.375       # Center of the 0.75-wide panel
        
        # 4. Toggle Button (Anchored to screen, not the panel, so it stays visible)
        self._toggle_btn = DirectButton(
            parent=self.a2dTopLeft,
            text="Close Menu",
            scale=0.04,
            pos=(0.16, 0, -0.06),
            frameSize=(-3.5, 3.5, -0.4, 0.8),
            frameColor=(0.2, 0.25, 0.3, 1.0),
            text_fg=(1, 1, 1, 1),
            command=self._toggle_ui_panel
        )

        # --- Build the UI using helpers ---
        
        self._add_ui_label("Control Panel", scale=0.06, is_title=True)
        
        self._add_ui_button("Toggle mouse / camera (F1)", self.toggle_mouse_camera_mode)
        self._add_ui_button("Pause / Resume", self._on_pause_resume)
        self._add_ui_button("Reset view", self._on_reset_view)
        self._add_ui_button("Reset simulation", self._on_reset_simulation)
        self._add_ui_button("Make step", self._on_make_step)
        
        self._add_ui_entry("Time Scale:", "1000", self._on_time_scale_change)
        self._add_ui_entry("Move speed:", "200", self._on_move_speed_change)

        # Add visual spacing before status text
        self._ui_current_z -= 0.05
        
        self._status_text = OnscreenText(
            parent=self._control_panel,
            text="Loading status...",
            pos=(self._ui_margin_x, self._ui_current_z),
            align=TextNode.ALeft,
            scale=0.038,
            fg=(0.95, 0.96, 1.0, 1.0),
            mayChange=True,
        )

    def _add_ui_label(self, text, scale=0.045, is_title=False):
        """Adds text to the panel and advances the vertical layout."""
        color = (1.0, 1.0, 1.0, 1.0) if is_title else (0.75, 0.75, 0.8, 1.0)
        lbl = OnscreenText(
            parent=self._control_panel,
            text=text,
            pos=(self._ui_margin_x, self._ui_current_z),
            align=TextNode.ALeft,
            scale=scale,
            fg=color,
        )
        self._ui_current_z -= (scale + 0.04) # Push next item down
        return lbl

    def _add_ui_button(self, text, command):
        """Adds a standardized button to the panel."""
        btn = DirectButton(
            parent=self._control_panel,
            text=text,
            scale=0.045,
            pos=(self._ui_center_x, 0, self._ui_current_z),
            frameSize=(-7.0, 7.0, -0.5, 0.75), # Fixed width based on text scale
            frameColor=(0.15, 0.18, 0.22, 1.0),
            text_fg=(0.9, 0.9, 0.9, 1.0),
            relief=2,
            command=command
        )
        self._ui_current_z -= 0.11
        return btn

    def _add_ui_entry(self, label_text, initial_value, command):
        """Adds a label and a text entry field beneath it."""
        self._add_ui_label(label_text, scale=0.04)
        entry = DirectEntry(
            parent=self._control_panel,
            text="",
            scale=0.045,
            pos=(self._ui_margin_x + 0.05, 0, self._ui_current_z),
            initialText=initial_value,
            numLines=1,
            focus=0,
            command=command,
            width=10,
            frameColor=(0.1, 0.1, 0.1, 1.0),
            text_fg=(1, 1, 1, 1)
        )
        self._ui_current_z -= 0.11
        return entry

    def _toggle_ui_panel(self):
        """Shows or hides the main side panel."""
        self.ui_is_open = not self.ui_is_open
        if self.ui_is_open:
            self._control_panel.show()
            self._toggle_btn['text'] = "Close Menu"
        else:
            self._control_panel.hide()
            self._toggle_btn['text'] = "Menu"

    def _on_pause_resume(self):
        self.pause = not self.pause

    def _on_reset_view(self):
        self.camera_controller.position = np.array([0, 0, 500], dtype=float)
        self.camera_controller.yaw = 0.
        self.camera_controller.pitch = -89.

    def _on_reset_simulation(self):
        pass

    def _on_make_step(self):
        if self.simulation is not None and self.pause:
            self.simulation.step()

    def _on_time_scale_change(self, text):
        try:
            value = float(text)
            if value <= 0:
                raise ValueError("Time scale must be positive.")
            self.sim_time_per_sec = value
        except ValueError:
            print(f"Invalid time scale input: '{text}'. Please enter a positive number.")

    def _on_move_speed_change(self, text):
        try:
            value = float(text)
            if value <= 0:
                raise ValueError("Move speed must be positive.")
            self.camera_controller.move_speed = value
        except ValueError:
            print(f"Invalid move speed input: '{text}'. Please enter a positive number.")
    
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
    
    def _create_brick_template(self, length: float = 2.0, width: float = 1.0, height: float = 0.5) -> NodePath:
        format_ = GeomVertexFormat.getV3n3()
        vertex_data = GeomVertexData("brick", format_, Geom.UHStatic)
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        normal_writer = GeomVertexWriter(vertex_data, "normal")

        # Half dimensions
        hx, hy, hz = length / 2.0, width / 2.0, height / 2.0

        # Define faces: Normal Vector, and 4 vertices in CCW order from the outside
        faces = [
            # Nose (+X): Looking from +X. Local Y is left/right, Local Z is up/down.
            (Vec3(1, 0, 0), [Vec3(hx, -hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(hx, -hy, hz)]),
            
            # Tail (-X): Looking from -X. Local Y is right/left, Local Z is up/down.
            (Vec3(-1, 0, 0), [Vec3(-hx, hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-hx, -hy, hz), Vec3(-hx, hy, hz)]),
            
            # Right side (+Y): Looking from +Y. Local X is left/right, Local Z is up/down.
            (Vec3(0, 1, 0), [Vec3(-hx, hy, -hz), Vec3(-hx, hy, hz), Vec3(hx, hy, hz), Vec3(hx, hy, -hz)]),
            
            # Left side (-Y): Looking from -Y. Local X is right/left, Local Z is up/down.
            (Vec3(0, -1, 0), [Vec3(-hx, -hy, -hz), Vec3(hx, -hy, -hz), Vec3(hx, -hy, hz), Vec3(-hx, -hy, hz)]),
            
            # Bottom (+Z): Looking from +Z. Local X is left/right, Local Y is up/down.
            (Vec3(0, 0, 1), [Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz)]),
            
            # Top (-Z): Looking from -Z. Local X is right/left, Local Y is up/down.
            (Vec3(0, 0, -1), [Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, -hy, -hz), Vec3(-hx, -hy, -hz)])
        ]

        triangles = GeomTriangles(Geom.UHStatic)
        vertex_index = 0

        for normal, verts in faces:
            for v in verts:
                vertex_writer.addData3f(v)
                normal_writer.addData3f(normal)
            
            # Add the two triangles per face using the CCW order
            triangles.addVertices(vertex_index, vertex_index + 1, vertex_index + 2)
            triangles.closePrimitive()
            triangles.addVertices(vertex_index, vertex_index + 2, vertex_index + 3)
            triangles.closePrimitive()
            
            vertex_index += 4

        geom = Geom(vertex_data)
        geom.addPrimitive(triangles)
        node = GeomNode("brick-template")
        node.addGeom(geom)
        
        return NodePath(node)

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
    
    def simulation_task(self, task):
        if not self.pause:
            dt = globalClock.getDt()
            N = int(self.sim_time_per_sec * dt / self.simulation.dt)
            for _ in range(max(1, N)):
                self.simulation.step()
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
            pos = body.get_position() * SCALE

            if not hasattr(body, "node"):
                if isinstance(body, Spacecraft):
                    body.node = self._brick_template.copyTo(self.render)
                    body.node.setColor(0.8, 0.8, 0.8, 1.0)
                else:
                    body.node = self._sphere_template.copyTo(self.render)
                    if body.name.lower() == "earth":
                        body.node.setColor(0.35, 0.65, 1.0, 1.0)
                    else:
                        body.node.setColor(1.0, 0.95, 0.75, 1.0)
                
                # body.node.setLightOff() 

            size = 1000 if isinstance(body, Spacecraft) else body.radius
            
            body_scale = max(size * SCALE, 1)
            body.node.setScale(body_scale)
            body.node.setPos(*pos)
            body.node.setQuat(Quat(*tuple(body.get_attitude())))