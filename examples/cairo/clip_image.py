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

cr.arc(128.0, 128.0, 76.8, 0, 2 * math.pi)
cr.clip()
cr.new_path()  # path not consumed by clip

# Source image is from:
# - https://www.publicdomainpictures.net/en/view-image.php?image=7683&picture=breaking-blue-wave
# Converted to PNG before using it
image = cairo.ImageSurface.create_from_png("examples/cairo/breaking-blue-wave.png")
w = image.get_width()
h = image.get_height()

cr.scale(256.0 / w, 256.0 / h)
cr.set_source_surface(image, 0, 0)
cr.paint()

# Save the image surface to a PNG file
surface.write_to_png("cairo-clip-image.png")
