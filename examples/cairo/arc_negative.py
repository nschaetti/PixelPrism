import cairo
import math

# Variables
xc = 128.0
yc = 128.0
radius = 100.0
angle1 = 45.0 * (math.pi / 180.0)  # angles are specified in radians
angle2 = 180.0 * (math.pi / 180.0)

# Create a surface and context
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 256, 256)
cr = cairo.Context(surface)

# Set line width and draw negative arc
cr.set_line_width(10.0)
cr.arc_negative(xc, yc, radius, angle1, angle2)
cr.stroke()

# Draw helping lines
cr.set_source_rgba(1, 0.2, 0.2, 0.6)
cr.set_line_width(6.0)

cr.arc(xc, yc, 10.0, 0, 2 * math.pi)
cr.fill()

cr.arc(xc, yc, radius, angle1, angle1)
cr.line_to(xc, yc)
cr.arc(xc, yc, radius, angle2, angle2)
cr.line_to(xc, yc)
cr.stroke()

# Save the result to a file
surface.write_to_png('arc_negative_example.png')
