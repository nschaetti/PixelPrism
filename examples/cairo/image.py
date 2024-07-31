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

cr.translate(128.0, 128.0)
cr.rotate(45 * math.pi / 180)
cr.scale(256.0 / w, 256.0 / h)
cr.translate(-0.5 * w, -0.5 * h)

cr.set_source_surface(image, 0, 0)
cr.paint()

# Save the image surface to a PNG file
surface.write_to_png("cairo-image.png")
