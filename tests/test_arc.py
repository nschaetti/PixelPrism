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
import unittest
import math
import numpy as np
from unittest.mock import Mock
from pixel_prism import p2, s, c
from pixel_prism.data import Point2D, Scalar, Color
from pixel_prism.drawing import Arc


class TestArc(unittest.TestCase):

    def test_initialization(self):
        """
        Test that the arc is initialized correctly.
        """
        arc = Arc.from_objects(
            center=p2(100, 100),
            radius=s(50),
            start_angle=s(0),
            end_angle=s(math.pi/2),
            line_width=s(2.0),
            line_color=c('RED'),
            fill_color=c('GREEN')
        )

        self.assertEqual(arc.cx, 100)
        self.assertEqual(arc.cy, 100)
        self.assertEqual(arc.radius.value, 50)
        self.assertEqual(arc.start_angle.value, 0)
        self.assertEqual(arc.end_angle.value, math.pi / 2)
        self.assertEqual(arc.line_width.value, 2.0)
        self.assertEqual(arc.line_color, Color(255, 0, 0))
        self.assertEqual(arc.fill_color, Color(0, 255, 0))
    # end test_initialization

    def test_properties(self):
        """
        Test that the properties of the arc can be set.
        """
        arc = Arc.from_objects(p2(0, 0), s(0), s(50), s(0), s(math.pi))
        arc.cx = 200
        self.assertEqual(arc.cx, 200)
        arc.cy = 150
        self.assertEqual(arc.cy, 150)
        arc.radius = Scalar(75)
        self.assertEqual(arc.radius.value, 75)
        arc.start_angle = Scalar(math.pi / 4)
        self.assertEqual(arc.start_angle.value, math.pi / 4)
        arc.end_angle = Scalar(3 * math.pi / 4)
        self.assertEqual(arc.end_angle.value, 3 * math.pi / 4)
    # end test_properties

    def test_update_points(self):
        """
        Test that the points of the arc are updated correctly.
        """
        arc = Arc.from_objects(p2(0, 0), s(50), s(0), s(math.pi / 2))
        self.assertAlmostEqual(arc.start_point.x, 50)
        self.assertAlmostEqual(arc.start_point.y, 0)
        self.assertAlmostEqual(arc.end_point.x, 0)
        self.assertAlmostEqual(arc.end_point.y, 50)
        self.assertAlmostEqual(arc.middle_point.x, 35.35533905932738)
        self.assertAlmostEqual(arc.middle_point.y, 35.35533905932738)
    # end test_update_points

    def test_update_bbox(self):
        """
        Test that the bounding box of the arc is updated correctly.
        """
        arc = Arc.from_objects(p2(0, 0), s(50), s(0), s(math.pi / 2))
        self.assertAlmostEqual(arc.bounding_box.upper_left.x, 0)
        self.assertAlmostEqual(arc.bounding_box.upper_left.y, 0)
        self.assertAlmostEqual(arc.bounding_box.width, 50)
        self.assertAlmostEqual(arc.bounding_box.height, 50)
    # end test_update_bbox

    def test_translating(self):
        """
        Test that the arc can be translated.
        """
        arc = Arc(p2(0, 0), s(50), s(0), s(math.pi / 2))
        arc.translate(p2(10, 15))
        self.assertEqual(arc.cx, 10)
        self.assertEqual(arc.cy, 15)
        self.assertAlmostEqual(arc.start_point.x, arc.cx + 50)
        self.assertAlmostEqual(arc.start_point.y, arc.cy)
    # end test_translating

    def test_events(self):
        """
        Test that the on_change event is triggered when the properties of the arc are modified.
        """
        arc = Arc.from_objects(p2(0, 0), s(50),  s(0), s(math.pi / 2))
        mock_on_change = Mock()
        arc.center.add_event_listener("on_change", mock_on_change)
        arc.cx = 100
        arc.cy = 200
        mock_on_change.assert_called()
        self.assertEqual(mock_on_change.call_count, 2)
    # end test_events

    def test_draw(self):
        """
        Test that the arc can be drawn.
        """
        arc = Arc(p2(0, 0), s(50), s(0), s(math.pi / 2), line_width=s(2.0), line_color=c('RED'))
        context = Mock()
        arc.draw(context)
        context.arc.assert_called()
        context.set_line_width.assert_called_with(2.0)
        context.set_source_rgba.assert_called()  # This checks that color setting function is called
        context.stroke.assert_called()
    # end test_draw

    def test_draw_points(self):
        """
        Test that the points of the arc are drawn.
        """
        arc = Arc(p2(0, 0), s(50), s(0), s(math.pi / 2))
        context = Mock()
        arc.draw_points(context)
        self.assertEqual(context.arc.call_count, 4)  # start point, end point, middle point, center
        context.stroke.assert_called()
    # end test_draw_points

    def test_from_scalar(self):
        """
        Test that an arc can be created from scalar values.
        """
        arc = Arc.from_scalar(
            center_x=50,
            center_y=50,
            radius=100,
            start_angle=0,
            end_angle=math.pi
        )
        self.assertEqual(arc.cx, 50)
        self.assertEqual(arc.cy, 50)
        self.assertEqual(arc.radius.value, 100)
        self.assertEqual(arc.start_angle.value, 0)
        self.assertEqual(arc.end_angle.value, math.pi)
    # end test_from_scalar

# end TestArc


if __name__ == '__main__':
    unittest.main()
# end if
