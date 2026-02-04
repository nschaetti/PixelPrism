import moderngl
import numpy as np
import cairo
import glm

WIDTH, HEIGHT = 1920, 1080
GL_SIZE = 800
OUTPUT = "cube_cairo.png"

ctx = moderngl.create_standalone_context()

color = ctx.texture((GL_SIZE, GL_SIZE), 4)
depth = ctx.depth_texture((GL_SIZE, GL_SIZE))
fbo = ctx.framebuffer(color, depth)
fbo.use()

ctx.viewport = (0, 0, GL_SIZE, GL_SIZE)
ctx.enable(moderngl.DEPTH_TEST)
ctx.clear(0, 0, 0, 0, depth=1.0)

prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec3 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        uniform mat4 mvp;
        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
            v_color = in_color;
        }
    """,
    fragment_shader="""
        #version 330
        in vec3 v_color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(v_color, 1.0);
        }
    """
)

S = 0.5
vertices = np.array([
    # front (red)
    -S,-S, S, 1,0,0,  S,-S, S, 1,0,0,  S, S, S, 1,0,0,
    -S,-S, S, 1,0,0,  S, S, S, 1,0,0, -S, S, S, 1,0,0,

    # right (green)
     S,-S, S, 0,1,0,  S,-S,-S, 0,1,0,  S, S,-S, 0,1,0,
     S,-S, S, 0,1,0,  S, S,-S, 0,1,0,  S, S, S, 0,1,0,

    # top (blue)
    -S, S, S, 0,0,1,  S, S, S, 0,0,1,  S, S,-S, 0,0,1,
    -S, S, S, 0,0,1,  S, S,-S, 0,0,1, -S, S,-S, 0,0,1,
], dtype="f4")

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])

proj = glm.perspective(glm.radians(45), 1.0, 0.1, 100.0)
view = glm.lookAt(glm.vec3(3, 3, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
model = glm.mat4(1)

prog["mvp"].write(bytes(proj * view * model))

vao.render()

raw = fbo.read(components=4)
img = np.frombuffer(raw, dtype=np.uint8).reshape(GL_SIZE, GL_SIZE, 4)
img = np.flipud(img).copy()
img = np.ascontiguousarray(img[:, :, [3, 0, 1, 2]])

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
cr = cairo.Context(surface)
cr.set_source_rgb(0, 0, 0)
cr.paint()

gl_surface = cairo.ImageSurface.create_for_data(
    img, cairo.FORMAT_ARGB32, GL_SIZE, GL_SIZE, GL_SIZE * 4
)

cr.set_source_surface(
    gl_surface,
    WIDTH - GL_SIZE,
    (HEIGHT - GL_SIZE) // 2
)
cr.paint()

surface.write_to_png(OUTPUT)
print("OK:", OUTPUT)
