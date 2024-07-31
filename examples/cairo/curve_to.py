import cairo

# Create a new image surface
width = 256
height = 256
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

# Create a Cairo Context for the surface
cr = cairo.Context(surface)
cr.set_source_rgb(0.95, 0.95, 0.95)
cr.paint()

# The main code
x = 25.6
y = 128.0
x1 = 102.4
y1 = 230.4
x2 = 153.6
y2 = 25.6
x3 = 230.4
y3 = 128.0

cr.set_source_rgb(0, 0, 0)
cr.move_to(x, y)
cr.curve_to(x1, y1, x2, y2, x3, y3)
cr.set_line_width(10.0)
cr.stroke()

cr.set_source_rgba(1, 0.2, 0.2, 0.6)
cr.set_line_width(6.0)
cr.move_to(x, y)
cr.line_to(x1, y1)
cr.move_to(x2, y2)
cr.line_to(x3, y3)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-curve-to.png")
