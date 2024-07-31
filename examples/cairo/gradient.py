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
cr.set_source_rgb(0, 0, 0)

# Create the Linear Pattern
linear_pattern = cairo.LinearGradient(0.0, 0.0, 0.0, 256.0)
linear_pattern.add_color_stop_rgba(1, 0, 0, 0, 1)
linear_pattern.add_color_stop_rgba(0, 1, 1, 1, 1)
cr.rectangle(0, 0, 256, 256)
cr.set_source(linear_pattern)
cr.fill()

# Create the Radial Pattern
radial_pattern = cairo.RadialGradient(115.2, 102.4, 25.6, 102.4, 102.4, 128.0)
radial_pattern.add_color_stop_rgba(0, 1, 1, 1, 1)
radial_pattern.add_color_stop_rgba(1, 0, 0, 0, 1)
cr.set_source(radial_pattern)

# Draw the circle filled with the radial pattern
cr.arc(128.0, 128.0, 76.8, 0, 2 * math.pi)
cr.fill()

# Save the image surface to a PNG file
surface.write_to_png("cairo-gradient.png")
