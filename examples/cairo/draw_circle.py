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

# Method to draw a zero
def draw_zero(context):
    # Define the path
    context.new_path()
    context.move_to(150, 100)
    context.arc(100, 100, 50, 0, 2 * 3.14159)
    context.new_sub_path()
    context.move_to(130, 100)
    context.arc(100, 100, 30, 0, 2 * 3.14159)

    # Fill the path
    context.set_source_rgb(0, 0, 1)  # Bleu
    context.fill()
# end draw_zero

# Create a surface and a context
width, height = 200, 200
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
context = cairo.Context(surface)

context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

# Set the drawing color
context.set_antialias(cairo.ANTIALIAS_DEFAULT)

# Draw a zero
draw_zero(context)

# Write the surface to a PNG file
surface.write_to_png('zero.png')

