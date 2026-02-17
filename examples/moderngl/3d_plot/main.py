"""Standalone ModernGL example: stylized mathematical 3D surface.

Dependencies:
    pip install moderngl glfw numpy

Run:
    python examples/moderngl/3d_plot/main.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import glfw
import moderngl
import numpy as np


PALETTES = ("scientific", "viridis", "magma", "inferno", "turbo")
PALETTE_TO_ID = {name: idx for idx, name in enumerate(PALETTES)}


def perspective_matrix(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_vec = np.cross(right, forward)
    up_vec = up_vec / np.linalg.norm(up_vec)

    m = np.eye(4, dtype="f4")
    m[0, :3] = right
    m[1, :3] = up_vec
    m[2, :3] = -forward
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(up_vec, eye)
    m[2, 3] = np.dot(forward, eye)
    return m


def mat4_to_bytes(matrix: np.ndarray) -> bytes:
    return matrix.astype("f4").T.tobytes()


@dataclass
class Camera:
    width: int
    height: int

    def __post_init__(self) -> None:
        self.eye = np.array([7.0, -6.0, 5.0], dtype="f4")
        self.target = np.array([0.0, 0.0, 0.0], dtype="f4")
        self.up = np.array([0.0, 0.0, 1.0], dtype="f4")
        self.min_radius = 5.0
        self.max_radius = 60.0

        offset = self.eye - self.target
        self.radius = float(np.linalg.norm(offset))
        self.yaw_deg = math.degrees(math.atan2(float(offset[1]), float(offset[0])))
        horiz = math.sqrt(float(offset[0] * offset[0] + offset[1] * offset[1]))
        self.pitch_deg = math.degrees(math.atan2(float(offset[2]), horiz))

        self.model = np.eye(4, dtype="f4")
        self.projection = np.eye(4, dtype="f4")
        self.view = np.eye(4, dtype="f4")
        self.mvp = np.eye(4, dtype="f4")
        self.update_matrices(self.width, self.height)

    def update_orbit_position(self) -> None:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        horiz = self.radius * math.cos(pitch)
        x = self.target[0] + horiz * math.cos(yaw)
        y = self.target[1] + horiz * math.sin(yaw)
        z = self.target[2] + self.radius * math.sin(pitch)
        self.eye = np.array([x, y, z], dtype="f4")

    def orbit(self, delta_yaw_deg: float, delta_pitch_deg: float) -> None:
        self.yaw_deg += delta_yaw_deg
        self.pitch_deg = float(np.clip(self.pitch_deg + delta_pitch_deg, -80.0, 80.0))
        self.update_orbit_position()

    def zoom(self, scroll_delta: float) -> None:
        factor = math.exp(-scroll_delta * 0.12)
        self.radius = float(np.clip(self.radius * factor, self.min_radius, self.max_radius))
        self.update_orbit_position()

    def update_matrices(self, width: int, height: int) -> None:
        self.width = max(width, 1)
        self.height = max(height, 1)
        aspect = self.width / self.height
        self.projection = perspective_matrix(45.0, aspect, 0.1, 200.0)
        self.update_orbit_position()
        self.view = look_at_matrix(self.eye, self.target, self.up)
        self.mvp = self.projection @ self.view @ self.model


class Grid:
    def __init__(
        self,
        ctx: moderngl.Context,
        *,
        mvp: np.ndarray,
        size_x: float,
        size_y: float,
        step: float,
        z_level: float,
        color: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float],
        border_thickness: float,
    ) -> None:
        self.ctx = ctx
        self.base_line_width = 1.0
        self.border_thickness = max(border_thickness, 0.001)
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = v_color;
                }
            """,
        )

        vertices = self._build_vertices(size_x=size_x, size_y=size_y, step=step, z_level=z_level, color=color)
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 4f", "in_pos", "in_color")])

        border_vertices = self._build_border_vertices(
            size_x=size_x,
            size_y=size_y,
            z_level=z_level,
            thickness=self.border_thickness,
            color=border_color,
        )
        self.border_vbo = self.ctx.buffer(border_vertices.tobytes())
        self.border_vao = self.ctx.vertex_array(self.prog, [(self.border_vbo, "3f 4f", "in_pos", "in_color")])
        self.set_mvp(mvp)

    @staticmethod
    def _build_vertices(
        *,
        size_x: float,
        size_y: float,
        step: float,
        z_level: float,
        color: tuple[float, float, float, float],
    ) -> np.ndarray:
        half_x = size_x * 0.5
        half_y = size_y * 0.5
        vals_x = np.arange(-half_x, half_x + 0.0001, step, dtype="f4")
        vals_y = np.arange(-half_y, half_y + 0.0001, step, dtype="f4")
        lines: list[list[float]] = []

        def add_vertex(px: float, py: float, pz: float) -> None:
            lines.append([px, py, pz, color[0], color[1], color[2], color[3]])

        for x in vals_x:
            add_vertex(x, -half_y, z_level)
            add_vertex(x, half_y, z_level)
        for y in vals_y:
            add_vertex(-half_x, y, z_level)
            add_vertex(half_x, y, z_level)

        return np.array(lines, dtype="f4")

    @staticmethod
    def _build_border_vertices(
        *,
        size_x: float,
        size_y: float,
        z_level: float,
        thickness: float,
        color: tuple[float, float, float, float],
    ) -> np.ndarray:
        half_x = size_x * 0.5
        half_y = size_y * 0.5
        t = min(thickness, half_x * 0.5, half_y * 0.5)
        c = [color[0], color[1], color[2], color[3]]

        tris: list[list[float]] = []

        def add_rect(x0: float, y0: float, x1: float, y1: float) -> None:
            p0 = [x0, y0, z_level]
            p1 = [x1, y0, z_level]
            p2 = [x1, y1, z_level]
            p3 = [x0, y1, z_level]
            tris.append([p0[0], p0[1], p0[2], *c])
            tris.append([p1[0], p1[1], p1[2], *c])
            tris.append([p2[0], p2[1], p2[2], *c])
            tris.append([p0[0], p0[1], p0[2], *c])
            tris.append([p2[0], p2[1], p2[2], *c])
            tris.append([p3[0], p3[1], p3[2], *c])

        add_rect(-half_x, half_y - t, half_x, half_y)
        add_rect(-half_x, -half_y, half_x, -half_y + t)
        add_rect(-half_x, -half_y + t, -half_x + t, half_y - t)
        add_rect(half_x - t, -half_y + t, half_x, half_y - t)

        return np.array(tris, dtype="f4")

    def set_mvp(self, mvp: np.ndarray) -> None:
        self.prog["mvp"].write(mat4_to_bytes(mvp))

    def render(self) -> None:
        self.ctx.line_width = self.base_line_width
        self.vao.render(mode=moderngl.LINES)
        self.border_vao.render(mode=moderngl.TRIANGLES)


class AxisLines:
    def __init__(
        self,
        ctx: moderngl.Context,
        *,
        mvp: np.ndarray,
        length: float,
        thickness: float,
        x_color: tuple[float, float, float, float],
        y_color: tuple[float, float, float, float],
        z_color: tuple[float, float, float, float],
    ) -> None:
        self.ctx = ctx
        self.thickness = max(thickness, 1.0)
        half = length * 0.5

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = v_color;
                }
            """,
        )

        vertices = np.array(
            [
                [-half, 0.0, 0.0, x_color[0], x_color[1], x_color[2], x_color[3]],
                [half, 0.0, 0.0, x_color[0], x_color[1], x_color[2], x_color[3]],
                [0.0, -half, 0.0, y_color[0], y_color[1], y_color[2], y_color[3]],
                [0.0, half, 0.0, y_color[0], y_color[1], y_color[2], y_color[3]],
                [0.0, 0.0, -half, z_color[0], z_color[1], z_color[2], z_color[3]],
                [0.0, 0.0, half, z_color[0], z_color[1], z_color[2], z_color[3]],
            ],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 4f", "in_pos", "in_color")])
        self.set_mvp(mvp)

    def set_mvp(self, mvp: np.ndarray) -> None:
        self.prog["mvp"].write(mat4_to_bytes(mvp))

    def render(self) -> None:
        self.ctx.line_width = self.thickness
        self.vao.render(mode=moderngl.LINES)


class SurfaceMesh:
    def __init__(
        self,
        ctx: moderngl.Context,
        *,
        mvp: np.ndarray,
        domain_size: float = 24.0,
        resolution: int = 90,
        palette_name: str = "scientific",
    ) -> None:
        self.ctx = ctx

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                in float in_face_z;
                flat out vec3 v_normal;
                flat out float v_z;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_normal = in_normal;
                    v_z = in_face_z;
                }
            """,
            fragment_shader="""
                #version 330
                flat in vec3 v_normal;
                flat in float v_z;
                out vec4 fragColor;
                uniform float zmin;
                uniform float zmax;
                uniform int palette_id;

                vec3 gradient5(float t, vec3 c0, vec3 c1, vec3 c2, vec3 c3, vec3 c4) {
                    if (t < 0.25) {
                        float k = t / 0.25;
                        return mix(c0, c1, k);
                    }
                    if (t < 0.50) {
                        float k = (t - 0.25) / 0.25;
                        return mix(c1, c2, k);
                    }
                    if (t < 0.75) {
                        float k = (t - 0.50) / 0.25;
                        return mix(c2, c3, k);
                    }
                    float k = (t - 0.75) / 0.25;
                    return mix(c3, c4, k);
                }

                vec3 apply_palette(float t, int pid) {
                    if (pid == 0) {
                        return gradient5(
                            t,
                            vec3(0.35, 0.10, 0.70),
                            vec3(0.10, 0.25, 0.90),
                            vec3(0.10, 0.90, 0.95),
                            vec3(0.97, 0.88, 0.20),
                            vec3(1.00, 1.00, 1.00)
                        );
                    }
                    if (pid == 1) {
                        return gradient5(
                            t,
                            vec3(0.267, 0.005, 0.329),
                            vec3(0.283, 0.141, 0.458),
                            vec3(0.254, 0.265, 0.530),
                            vec3(0.207, 0.520, 0.553),
                            vec3(0.993, 0.906, 0.144)
                        );
                    }
                    if (pid == 2) {
                        return gradient5(
                            t,
                            vec3(0.001, 0.000, 0.014),
                            vec3(0.201, 0.041, 0.414),
                            vec3(0.493, 0.118, 0.513),
                            vec3(0.873, 0.288, 0.408),
                            vec3(0.988, 0.998, 0.645)
                        );
                    }
                    if (pid == 3) {
                        return gradient5(
                            t,
                            vec3(0.001, 0.000, 0.014),
                            vec3(0.188, 0.039, 0.329),
                            vec3(0.472, 0.111, 0.428),
                            vec3(0.865, 0.318, 0.226),
                            vec3(0.988, 0.998, 0.645)
                        );
                    }
                    return gradient5(
                        t,
                        vec3(0.190, 0.071, 0.232),
                        vec3(0.270, 0.340, 0.706),
                        vec3(0.125, 0.733, 0.722),
                        vec3(0.898, 0.873, 0.177),
                        vec3(0.980, 0.729, 0.221)
                    );
                }

                void main() {
                    float denom = max(zmax - zmin, 1e-6);
                    float t = clamp((v_z - zmin) / denom, 0.0, 1.0);
                    vec3 color = apply_palette(t, palette_id);
                    float facet = 0.90 + 0.10 * abs(normalize(v_normal).z);
                    fragColor = vec4(color * facet, 1.0);
                }
            """,
        )

        self.wire_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                uniform vec4 wire_color;
                void main() {
                    fragColor = wire_color;
                }
            """,
        )

        vertices, indices, wire_vertices, wire_indices, zmin, zmax = self._build_mesh(
            domain_size=domain_size,
            resolution=resolution,
        )
        self.zmin = zmin
        self.zmax = zmax

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 1f", "in_pos", "in_normal", "in_face_z")],
            self.ibo,
        )

        self.wire_vbo = self.ctx.buffer(wire_vertices.tobytes())
        self.wire_ibo = self.ctx.buffer(wire_indices.tobytes())
        self.wire_vao = self.ctx.vertex_array(
            self.wire_prog,
            [(self.wire_vbo, "3f", "in_pos")],
            self.wire_ibo,
        )

        self.set_mvp(mvp)
        self.prog["zmin"].value = self.zmin
        self.prog["zmax"].value = self.zmax
        # self.wire_prog["wire_color"].value = (0.92, 0.92, 0.96, 0.10)
        self.wire_prog["wire_color"].value = (0.1, 0.1, 0.1, 0.01)
        self.set_palette(palette_name)

    @staticmethod
    def _height(px: float, py: float) -> float:
        r = math.sqrt(px * px + py * py)
        return math.sin(r) / (1.0 + 0.3 * r)

    def _build_mesh(
        self,
        *,
        domain_size: float,
        resolution: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        resolution = max(2, resolution)
        half = domain_size * 0.5

        xs = np.linspace(-half, half, resolution, dtype="f4")
        ys = np.linspace(-half, half, resolution, dtype="f4")

        z_samples = np.zeros((resolution, resolution), dtype="f4")
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                z_samples[iy, ix] = self._height(float(x), float(y))

        zmin = float(z_samples.min())
        zmax = float(z_samples.max())

        grid_vertices = np.zeros((resolution, resolution, 3), dtype="f4")
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                grid_vertices[iy, ix, 0] = x
                grid_vertices[iy, ix, 1] = y
                grid_vertices[iy, ix, 2] = z_samples[iy, ix]

        packed_vertices: list[list[float]] = []
        packed_indices: list[int] = []
        wire_indices: list[int] = []

        def point(ix: int, iy: int) -> np.ndarray:
            return grid_vertices[iy, ix]

        def emit_triangle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> None:
            edge1 = p1 - p0
            edge2 = p2 - p0
            normal = np.cross(edge1, edge2)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                normal = np.array([0.0, 0.0, 1.0], dtype="f4")
            else:
                normal = normal / norm

            face_z = float((p0[2] + p1[2] + p2[2]) / 3.0)
            base = len(packed_vertices)
            packed_vertices.append([p0[0], p0[1], p0[2], normal[0], normal[1], normal[2], face_z])
            packed_vertices.append([p1[0], p1[1], p1[2], normal[0], normal[1], normal[2], face_z])
            packed_vertices.append([p2[0], p2[1], p2[2], normal[0], normal[1], normal[2], face_z])
            packed_indices.extend([base, base + 1, base + 2])

        for iy in range(resolution - 1):
            for ix in range(resolution - 1):
                p00 = point(ix, iy)
                p10 = point(ix + 1, iy)
                p01 = point(ix, iy + 1)
                p11 = point(ix + 1, iy + 1)

                emit_triangle(p00, p10, p11)
                emit_triangle(p00, p11, p01)

        for iy in range(resolution):
            for ix in range(resolution - 1):
                a = iy * resolution + ix
                b = a + 1
                wire_indices.extend([a, b])

        for iy in range(resolution - 1):
            for ix in range(resolution):
                a = iy * resolution + ix
                b = (iy + 1) * resolution + ix
                wire_indices.extend([a, b])

        vertices = np.array(packed_vertices, dtype="f4")
        indices = np.array(packed_indices, dtype="u4")
        wire_vertices = grid_vertices.reshape(-1, 3).astype("f4")
        wire_indices_np = np.array(wire_indices, dtype="u4")
        return vertices, indices, wire_vertices, wire_indices_np, zmin, zmax

    def set_mvp(self, mvp: np.ndarray) -> None:
        self.prog["mvp"].write(mat4_to_bytes(mvp))
        self.wire_prog["mvp"].write(mat4_to_bytes(mvp))

    def set_palette(self, palette_name: str) -> None:
        if palette_name not in PALETTE_TO_ID:
            raise ValueError(f"Unknown palette '{palette_name}'. Available: {', '.join(PALETTES)}")
        self.palette_name = palette_name
        self.prog["palette_id"].value = PALETTE_TO_ID[palette_name]

    def cycle_palette(self, step: int = 1) -> str:
        idx = PALETTE_TO_ID[self.palette_name]
        next_idx = (idx + step) % len(PALETTES)
        next_name = PALETTES[next_idx]
        self.set_palette(next_name)
        return next_name

    def render(self) -> None:
        self.vao.render(mode=moderngl.TRIANGLES)
        self.wire_vao.render(mode=moderngl.LINES)


class StaticPlotApp:
    def __init__(self, width: int = 1280, height: int = 720, palette: str = "scientific") -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 8)

        self.window = glfw.create_window(width, height, "ModernGL Surface Plot", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.line_width = 1.0

        self.is_dragging = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.rotate_sensitivity = 0.22

        self.camera = Camera(width, height)

        grid_color = (0.82, 0.82, 0.82, 0.38)
        grid_border_color = (1.0, 1.0, 1.0, 0.88)
        grid_border_thickness = 0.02
        axis_length = 10.0
        axis_thickness = 2.0
        axis_x_color = (1.0, 0.28, 0.28, 1.0)
        axis_y_color = (0.28, 1.0, 0.42, 1.0)
        axis_z_color = (0.32, 0.58, 1.0, 1.0)

        self.grid = Grid(
            self.ctx,
            mvp=self.camera.mvp,
            size_x=10.0,
            size_y=10.0,
            step=1.0,
            z_level=0.0,
            color=grid_color,
            border_color=grid_border_color,
            border_thickness=grid_border_thickness,
        )
        self.axes = AxisLines(
            self.ctx,
            mvp=self.camera.mvp,
            length=axis_length,
            thickness=axis_thickness,
            x_color=axis_x_color,
            y_color=axis_y_color,
            z_color=axis_z_color,
        )
        self.surface = SurfaceMesh(
            self.ctx,
            mvp=self.camera.mvp,
            domain_size=8.0,
            resolution=20,
            palette_name=palette,
        )

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self._update_viewport(fb_w, fb_h)
        self._update_window_title()

    @staticmethod
    def _on_resize(window: glfw._GLFWwindow, width: int, height: int) -> None:
        app: StaticPlotApp = glfw.get_window_user_pointer(window)
        app._update_viewport(width, height)

    @staticmethod
    def _on_mouse_button(window: glfw._GLFWwindow, button: int, action: int, mods: int) -> None:
        del mods
        app: StaticPlotApp = glfw.get_window_user_pointer(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                app.is_dragging = True
                app.last_x, app.last_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                app.is_dragging = False

    @staticmethod
    def _on_cursor_move(window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        app: StaticPlotApp = glfw.get_window_user_pointer(window)
        if not app.is_dragging:
            return

        dx = xpos - app.last_x
        dy = ypos - app.last_y
        app.last_x = xpos
        app.last_y = ypos

        app.camera.orbit(dx * app.rotate_sensitivity, -dy * app.rotate_sensitivity)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        app._update_viewport(fb_w, fb_h)

    @staticmethod
    def _on_scroll(window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        del xoffset
        app: StaticPlotApp = glfw.get_window_user_pointer(window)
        app.camera.zoom(float(yoffset))
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        app._update_viewport(fb_w, fb_h)

    @staticmethod
    def _on_key(window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:
        del scancode
        del mods
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        app: StaticPlotApp = glfw.get_window_user_pointer(window)

        if key == glfw.KEY_C:
            app.surface.cycle_palette(1)
            app._update_window_title()
            return

        digit_map = {
            glfw.KEY_1: 0,
            glfw.KEY_2: 1,
            glfw.KEY_3: 2,
            glfw.KEY_4: 3,
            glfw.KEY_5: 4,
        }
        if key in digit_map:
            app.surface.set_palette(PALETTES[digit_map[key]])
            app._update_window_title()

    def _update_viewport(self, width: int, height: int) -> None:
        width = max(width, 1)
        height = max(height, 1)
        self.ctx.viewport = (0, 0, width, height)
        self.camera.update_matrices(width, height)
        mvp = self.camera.mvp
        self.grid.set_mvp(mvp)
        self.axes.set_mvp(mvp)
        self.surface.set_mvp(mvp)

    def render(self) -> None:
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.grid.render()

        self.ctx.disable(moderngl.BLEND)
        self.surface.render()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.axes.render()

    def _update_window_title(self) -> None:
        title = f"ModernGL Surface Plot - palette: {self.surface.palette_name} (C cycle, 1-5 select)"
        glfw.set_window_title(self.window, title)

    def run(self) -> None:
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.render()
                glfw.swap_buffers(self.window)
        finally:
            glfw.destroy_window(self.window)
            glfw.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModernGL 3D surface with selectable color palettes")
    parser.add_argument(
        "--palette",
        choices=PALETTES,
        default="scientific",
        help="Color palette for surface fill",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = StaticPlotApp(palette=args.palette)
    app.run()


if __name__ == "__main__":
    main()
