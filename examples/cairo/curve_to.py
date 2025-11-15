# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
