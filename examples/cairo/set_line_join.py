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
cr.set_line_width(40.96)

# First set of lines with LINE_JOIN_MITER
cr.move_to(76.8, 84.48)
cr.rel_line_to(51.2, -51.2)
cr.rel_line_to(51.2, 51.2)
cr.set_line_join(cairo.LINE_JOIN_MITER)  # default
cr.stroke()

# Second set of lines with LINE_JOIN_BEVEL
cr.move_to(76.8, 161.28)
cr.rel_line_to(51.2, -51.2)
cr.rel_line_to(51.2, 51.2)
cr.set_line_join(cairo.LINE_JOIN_BEVEL)
cr.stroke()

# Third set of lines with LINE_JOIN_ROUND
cr.move_to(76.8, 238.08)
cr.rel_line_to(51.2, -51.2)
cr.rel_line_to(51.2, 51.2)
cr.set_line_join(cairo.LINE_JOIN_ROUND)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-set-line-join.png")
