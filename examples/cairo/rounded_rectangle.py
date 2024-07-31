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
# A custom shape that could be wrapped in a function
x = 25.6        # parameters like cr.rectangle
y = 25.6
width = 204.8
height = 204.8
aspect = 1.0     # aspect ratio
corner_radius = height / 10.0   # and corner curvature radius

radius = corner_radius / aspect
degrees = math.pi / 180.0

cr.new_sub_path()
cr.arc(x + width - radius, y + radius, radius, -90 * degrees, 0 * degrees)
cr.arc(x + width - radius, y + height - radius, radius, 0 * degrees, 90 * degrees)
cr.arc(x + radius, y + height - radius, radius, 90 * degrees, 180 * degrees)
cr.arc(x + radius, y + radius, radius, 180 * degrees, 270 * degrees)
cr.close_path()

cr.set_source_rgb(0.5, 0.5, 1)
cr.fill_preserve()
cr.set_source_rgba(0.5, 0, 0, 0.5)
cr.set_line_width(10.0)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-rounded-rectangle.png")
