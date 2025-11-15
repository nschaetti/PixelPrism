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

# Imports
import cairo
import math

# Setup
WIDTH, HEIGHT = 400, 400


def draw_rectangle(context, color):
    """Draw a rectangle centered at the origin (0, 0) and set the color."""
    context.rectangle(-50, -50, 100, 100)  # Rectangle centered at (0, 0)
    context.set_source_rgb(*color)  # Set the color
    context.stroke()
# end draw_rectangle


with cairo.SVGSurface("transform_steps.png", WIDTH, HEIGHT) as surface:
    context = cairo.Context(surface)

    # Fill the background with white
    context.set_source_rgb(1, 1, 1)  # White background
    context.paint()  # Fill the entire surface with the current color

    # Set the line width for the drawings
    context.set_line_width(5)

    # --- Step 1: Draw the original rectangle ---
    # Original system (no transformation)
    context.save()  # Save context before any transformations
    context.set_source_rgb(0, 0, 0)  # Black color
    # context.rectangle(50, 50, 100, 100)  # Rectangle at (50, 50)
    draw_rectangle(context, (0, 0, 0))
    context.stroke()

    # --- Step 2: Apply translation ---
    context.restore()  # Restore the original context (before drawing)
    context.save()  # Save context for translation step
    context.translate(WIDTH / 2, HEIGHT / 2)  # Move origin to the center
    draw_rectangle(context, (0, 1, 0))  # Draw green rectangle after translation

    # --- Step 3: Apply rotation ---
    context.restore()  # Restore the original context
    context.save()  # Save context for rotation step
    context.translate(WIDTH / 2, HEIGHT / 2)  # Move origin to the center
    context.rotate(math.radians(45))  # Rotate by 45 degrees
    draw_rectangle(context, (0, 0, 1))  # Draw blue rectangle after rotation

    # --- Step 4: Apply scale ---
    context.restore()  # Restore the original context
    context.save()  # Save context for scaling step
    context.translate(WIDTH / 2, HEIGHT / 2)  # Move origin to the center
    context.rotate(math.radians(45))  # Rotate by 45 degrees
    context.scale(1.5, 1.5)  # Scale by 1.5
    draw_rectangle(context, (1, 0, 0))  # Draw red rectangle after scaling

    # Save the PNG file
    surface.write_to_png("transform_steps_with_background.png")
# end with

print("Example saved as 'transform_example.svg'")
