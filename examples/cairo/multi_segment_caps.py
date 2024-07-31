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
cr.move_to(50.0, 75.0)
cr.line_to(200.0, 75.0)

cr.move_to(50.0, 125.0)
cr.line_to(200.0, 125.0)

cr.move_to(50.0, 175.0)
cr.line_to(200.0, 175.0)

cr.set_line_width(30.0)
cr.set_line_cap(cairo.LINE_CAP_ROUND)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-multi-segment-caps.png")
