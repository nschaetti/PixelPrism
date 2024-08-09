#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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
cr.set_line_width(6)

cr.rectangle(12, 12, 232, 70)
cr.new_sub_path()
cr.arc(64, 64, 40, 0, 2 * math.pi)
cr.new_sub_path()
cr.arc_negative(192, 64, 40, 0, -2 * math.pi)

cr.set_fill_rule(cairo.FillRule.EVEN_ODD)
cr.set_source_rgb(0, 0.7, 0)
cr.fill_preserve()
cr.set_source_rgb(0, 0, 0)
cr.stroke()

cr.translate(0, 128)
cr.rectangle(12, 12, 232, 70)
cr.new_sub_path()
cr.arc(64, 64, 40, 0, 2 * math.pi)
cr.new_sub_path()
cr.arc_negative(192, 64, 40, 0, -2 * math.pi)

cr.set_fill_rule(cairo.FillRule.WINDING)
cr.set_source_rgb(0, 0, 0.9)
cr.fill_preserve()
cr.set_source_rgb(0, 0, 0)
cr.stroke()

# Save the image surface to a PNG file
surface.write_to_png("cairo-fill-style.png")
