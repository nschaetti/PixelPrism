
# This example demonstrates how to clip a drawing area using a circle and then
import cairo
import math

# Create a new image surface
width = 256
height = 256
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

# Create a Cairo Context for the surface
cr = cairo.Context(surface)
cr.set_source_rgb(1, 1, 1)
cr.paint()

# The main code
cr.set_source_rgb(0, 0, 0)
cr.arc(128.0, 128.0, 76.8, 0, 2 * math.pi)
cr.clip()

cr.new_path()  # current path is not affected by clip
cr.rectangle(0, 0, 256, 256)
cr.fill()

cr.set_source_rgb(0, 1, 0)
cr.move_to(0, 0)
cr.line_to(256, 256)
cr.move_to(256, 0)
cr.line_to(0, 256)
cr.set_line_width(10.0)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-clip.png")
