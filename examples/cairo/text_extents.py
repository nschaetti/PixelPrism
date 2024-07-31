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
utf8 = "cairo"

cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
cr.set_font_size(100.0)
extents = cr.text_extents(utf8)

x = 25.0
y = 150.0

cr.move_to(x, y)
cr.show_text(utf8)

# draw helping lines
cr.set_source_rgba(1, 0.2, 0.2, 0.6)
cr.set_line_width(6.0)
cr.arc(x, y, 10.0, 0, 2 * math.pi)
cr.fill()
cr.move_to(x, y)
cr.rel_line_to(0, -extents.height)
cr.rel_line_to(extents.width, 0)
cr.rel_line_to(extents.x_bearing, -extents.y_bearing)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-text-extents.png")
