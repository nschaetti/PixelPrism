import cairo
import math

# Create a new image surface
width = 256
height = 256
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

# Create a Cairo Context for the surface
cr = cairo.Context(surface)
cr.set_source_rgb(0.95, 0.95, 0.95)
cr.paint()

# The main code
image = cairo.ImageSurface.create_from_png("examples/cairo/breaking-blue-wave.png")
w = image.get_width()
h = image.get_height()

# Load the image as a surface pattern
pattern = cairo.SurfacePattern(image)
pattern.set_extend(cairo.EXTEND_REPEAT)

cr.translate(128.0, 128.0)
cr.rotate(math.pi / 4)
cr.scale(1 / math.sqrt(2), 1 / math.sqrt(2))
cr.translate(-128.0, -128.0)

# Set up the scale matrix
matrix = cairo.Matrix()
matrix.scale(w / 256.0 * 5.0, h / 256.0 * 5.0)
pattern.set_matrix(matrix)

cr.set_source(pattern)

cr.rectangle(0, 0, 256.0, 256.0)
cr.fill()

# Save the image surface to a PNG file
surface.write_to_png("cairo-image-pattern.png")
