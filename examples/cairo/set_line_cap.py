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
cr.set_source_rgb(0, 0, 0)
cr.set_line_width(30.0)
cr.set_line_cap(cairo.LINE_CAP_BUTT)  # default
cr.move_to(64.0, 50.0)
cr.line_to(64.0, 200.0)
cr.stroke()
cr.set_line_cap(cairo.LINE_CAP_ROUND)
cr.move_to(128.0, 50.0)
cr.line_to(128.0, 200.0)
cr.stroke()
cr.set_line_cap(cairo.LINE_CAP_SQUARE)
cr.move_to(192.0, 50.0)
cr.line_to(192.0, 200.0)
cr.stroke()

# draw helping lines
cr.set_source_rgb(1, 0.2, 0.2)
cr.set_line_width(2.56)
cr.move_to(64.0, 50.0)
cr.line_to(64.0, 200.0)
cr.move_to(128.0, 50.0)
cr.line_to(128.0, 200.0)
cr.move_to(192.0, 50.0)
cr.line_to(192.0, 200.0)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-set-line-cap.png")
