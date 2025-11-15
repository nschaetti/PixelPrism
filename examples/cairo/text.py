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

cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
cr.set_font_size(90.0)

cr.move_to(10.0, 135.0)
cr.show_text("Hello")

cr.move_to(70.0, 165.0)
cr.text_path("void")
cr.set_source_rgb(0.5, 0.5, 1)
cr.fill_preserve()
cr.set_source_rgb(0, 0, 0)
cr.set_line_width(2.56)
cr.stroke()

# draw helping lines
cr.set_source_rgba(1, 0.2, 0.2, 0.6)
cr.arc(10.0, 135.0, 5.12, 0, 2 * math.pi)
cr.close_path()
cr.arc(70.0, 165.0, 5.12, 0, 2 * math.pi)
cr.fill()

# Save the image surface to a PNG file
surface.write_to_png("cairo-text.png")
