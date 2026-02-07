

import moderngl
import numpy as np
import cairo
import glm

WIDTH, HEIGHT = 1920, 1080

GL_SIZE = 800

OUTPUT = "cube_cairo.png"

# Create a standalone ModernGL context 
ctx: moderngl.Context = moderngl.create_standalone_context()

# Create texture object.
# size: width and height of the texture => 800
# components (int): number of components
color = ctx.texture((GL_SIZE, GL_SIZE), 4)

# Create a texture object, juste size
depth = ctx.depth_texture((GL_SIZE, GL_SIZE))

# framebuffer is a collection of buffers that can 
# be used as the destination for rendering.
# TODO: we will render in color ?
fbo = ctx.framebuffer(color, depth)

# Bind the framebuffer. Set the target for VertexArray.render()
fbo.use()

# The viewport is the area of the framebuffer in which
# OpenGL will draw.
# Here, bottom: 0,0, width and height: 800x800
ctx.viewport = (0, 0, GL_SIZE, GL_SIZE)

# Enable DEPTH_TEXT => draw only if closer to view.
ctx.enable(moderngl.DEPTH_TEST)

# Transparent black, 0.0 : near, 1.0 far (infinity)
# In the depth buffer => infinity far.
ctx.clear(0, 0, 0, 0, depth=1.0)


prog = ctx.program(
    # Executé pour chaque sommet
    # in_pos : position du sommet
    # in_color: couleur du sommet 
    # v_color : données transmises au fragment shader 
    # Prends la position du sommet, transforme-le avec la matrice de la caméra,
    # et donne au GPU la position finale à l'écran.
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
    # Dernière étape avant l'écriture dans le framebuffer
    # in vec3 v_color; => v_color de vertex_shader
    # Chaque sommet a une couleur, le GPU interpole
    # out vec4 fragColor => ce qui est écrit dans le color buffer.
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
    # x, y, z, r, g, b  
    # Deux triangles par face
    # front (red)
    -S,-S, S, 1,0,0,  
     S,-S, S, 1,0,0,  
     S, S, S, 1,0,0,
    -S,-S, S, 1,0,0,  
     S, S, S, 1,0,0, 
    -S, S, S, 1,0,0,

    # right (green)
     S,-S, S, 0,1,0,  
     S,-S,-S, 0,1,0,  
     S, S,-S, 0,1,0,
     S,-S, S, 0,1,0,  
     S, S,-S, 0,1,0,  
     S, S, S, 0,1,0,

    # top (blue)
    -S, S, S, 0,0,1,  
     S, S, S, 0,0,1,  
     S, S,-S, 0,0,1,
    -S, S, S, 0,0,1,  
     S, S,-S, 0,0,1, 
    -S, S,-S, 0,0,1,
], dtype="f4")

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])

proj = glm.perspective(glm.radians(45), 1.0, 0.1, 100.0)
view = glm.lookAt(glm.vec3(3, 3, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
model = glm.mat4(1)

prog["mvp"].write(bytes(proj * view * model))

vao.render()

# Read the framebuffer 
raw = fbo.read(components=4)

# Convert framebuffer to numpy array 
img = np.frombuffer(raw, dtype=np.uint8).reshape(GL_SIZE, GL_SIZE, 4)

# flipud: reverse the order along axis 0 (upside down)
img = np.flipud(img).copy()

# Make it contiuguous in the memory
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
