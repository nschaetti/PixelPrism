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

# Imports
import cairo
import numpy as np
from lxml import etree

import svgpathtools as svg


# Parse path
def parse_path(
        path_data
):
    """
    Parse an SVG path and return the segments.

    Args:
        path_data (str): SVG path math_old
    """
    return svg.parse_path(path_data)
# end parse_path


# Parse SVG
def parse_svg(
        svg_path
):
    """
    Parse an SVG file and return the paths and transformations.

    Args:
        svg_path (str): Path to the SVG file
    """
    tree = etree.parse(svg_path)
    root = tree.getroot()

    # Keep track of paths and transformations
    defs = {}

    # Parse paths and transformations
    for element in root.findall('.//{http://www.w3.org/2000/svg}path'):
        path_id = element.attrib.get('id')
        path_data = element.attrib.get('d')
        defs[path_id] = svg.parse_path(path_data)
    # end for

    # Elements
    paths = []

    # Function to parse elements in order
    def parse_element(element):
        tag = etree.QName(element.tag).localname
        if tag == 'use':
            href = element.attrib.get('{http://www.w3.org/1999/xlink}href').strip('#')
            x = float(element.attrib.get('x', '0'))
            y = float(element.attrib.get('y', '0'))
            transform = element.attrib.get('transform', '')
            paths.append({'type': 'path', 'math_old': defs[href], 'x': x, 'y': y, 'transform': transform})
        elif tag == 'rect':
            x = float(element.attrib.get('x', '0'))
            y = float(element.attrib.get('y', '0'))
            width = float(element.attrib.get('width', '0'))
            height = float(element.attrib.get('height', '0'))
            transform = element.attrib.get('transform', '')
            paths.append({'type': 'rect', 'x': x, 'y': y, 'width': width, 'height': height, 'transform': transform})
        # end if
    # end parse_element

    # Traverse the root element to preserve order
    for element in root.iter():
        parse_element(element)
    # end for

    return paths
# end parse_svg


def apply_transform(context, transform):
    """
    Apply an SVG transform to a Cairo context.

    Args:
        context (cairo.Context): Context to apply the transform to
        transform (str): SVG transform
    """
    # Translate
    if transform.startswith('translate'):
        values = transform[10:-1].split(',')
        tx = float(values[0])
        ty = float(values[1]) if len(values) > 1 else 0.0
        context.translate(tx, ty)
    # Scale
    elif transform.startswith('scale'):
        values = transform[6:-1].split(',')
        sx = float(values[0])
        sy = float(values[1]) if len(values) > 1 else sx
        context.scale(sx, sy)
    # Rotate
    elif transform.startswith('rotate'):
        values = transform[7:-1].split(',')
        angle = float(values[0])
        cx = float(values[1]) if len(values) > 1 else 0.0
        cy = float(values[2]) if len(values) > 2 else 0.0
        context.translate(cx, cy)
        context.rotate(np.radians(angle))
        context.translate(-cx, -cy)
    elif transform.startswith('skewX'):
        angle = np.radians(float(transform[6:-1]))
        matrix = cairo.Matrix(1, np.tan(angle), 0, 1, 0, 0)
        context.transform(matrix)
    elif transform.startswith('skewY'):
        angle = np.radians(float(transform[6:-1]))
        matrix = cairo.Matrix(1, 0, np.tan(angle), 1, 0, 0)
        context.transform(matrix)
    elif transform.startswith('matrix'):
        values = transform[7:-1].split(',')
        matrix = cairo.Matrix(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5])
        )
        context.transform(matrix)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    # end if
# end apply_transform


# Draw SVG
def draw_svg(
        context,
        svg_path,
        x,
        y,
        color=(0, 0, 0)
):
    """
    Draw an SVG file to the context.

    Args:
        context (cairo.Context): Context to draw the SVG file to
        svg_path (str): Path to the SVG file
        x (int): X-coordinate to draw the SVG file
        y (int): Y-coordinate to draw the SVG file
        color (tuple): The color to use for the SVG (r, g, b)
    """
    # Parse the SVG file
    paths = parse_svg(svg_path)

    # Draw the SVG file
    context.save()
    context.translate(x, y)
    context.scale(10, 10)
    context.set_source_rgb(*color)  # Set the drawing color
    context.set_fill_rule(cairo.FILL_RULE_WINDING)
    context.set_line_width(0.2)

    # Draw the paths
    for el_i, element in enumerate(paths):
        # Save the context
        context.save()

        # Translate to position
        context.translate(element['x'], element['y'])

        # Apply transformations
        if element['transform']:
            apply_transform(context, element['transform'])
        # end if

        if element['type'] == 'path':
            # Get subpaths
            subpaths = element['math_old'].d().split('M')

            # New path
            context.new_path()

            # For each subpaths
            for subpath in subpaths:
                if not subpath.strip():
                    continue
                # end if

                # Move to the first point
                subpath = 'M' + subpath.strip()

                # Parse the subpath
                sub_path = svg.parse_path(subpath)

                # Draw the segments
                for segment in sub_path:
                    if isinstance(segment, svg.Line):
                        context.line_to(segment.end.real, segment.end.imag)
                    elif isinstance(segment, svg.CubicBezier):
                        context.curve_to(
                            segment.control1.real,
                            segment.control1.imag,
                            segment.control2.real,
                            segment.control2.imag,
                            segment.end.real,
                            segment.end.imag
                        )
                    elif isinstance(segment, svg.QuadraticBezier):
                        context.curve_to(
                            segment.control.real,
                            segment.control.imag,
                            segment.control.real,
                            segment.control.imag,
                            segment.end.real,
                            segment.end.imag
                        )
                    elif isinstance(segment, svg.Arc):
                        context.arc(
                            segment.center.real,
                            segment.center.imag,
                            segment.radius,
                            segment.start_angle,
                            segment.end_angle
                        )
                    # end if
                # end for
            # end for

            # Fill the path
            context.fill()
        elif element['type'] == 'rect':
            context.rectangle(0, 0, element['width'], element['height'])
            context.fill()
        # end if

        # Restore the context
        context.restore()
    # end for

    # Restore the context
    context.restore()
# end draw_svg
