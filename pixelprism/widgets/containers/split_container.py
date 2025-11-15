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

# Import necessary libraries.
from .container import Container


class SplitContainer(Container):
    """
    Split container class.
    """

    def __init__(
            self,
            size: int,
            orientation='horizontal'
    ):
        """
        Initialize the split container.

        Args:
            size (int): Size of the container
            orientation (str): Orientation of the container (horizontal or vertical)
        """
        super().__init__()
        self.size = size
        self.orientation = orientation
    # end __init__

    # Create a surface for a widget
    def create_surface(
            self,
            widget,
            **kwargs
    ):
        """
        Create a sub-surface for a widget.

        Args:
            widget (Widget): The widget to create a surface for.
            kwargs: Additional arguments (not used in this implementation)
        """
        # Get the widget's position and size
        position = kwargs.get("position")
        assert position is not None, "Position must be specified."

        # Compute position and size according to orientation and position
        if self.orientation == 'horizontal':
            # Compute position and size according to position
            x = (self.width // self.size) * position
            y = 0
            width = self.width // self.size
            height = self.height
        else:
            # vertical
            x = 0
            y = (self.height // self.size) * position
            width = self.width
            height = self.height // self.size
        # end if

        # Create a sub-surface for the widget
        return self.surface.create_for_rectangle(
            x,
            y,
            width,
            height
        )
    # end create_surface

    # Add a widget to the container
    def add_widget(
            self,
            widget,
            position: int
    ):
        """
        Add a widget to the container.

        Args:
            widget (Widget): The widget to add to the container.
            position (int): Zone of the widget
        """
        assert 0 <= position < self.size, f"Position must be between 0 and {self.size - 1}."
        if self._check_zone(position):
            raise ValueError(f"Position {position} is already occupied.")
        else:
            super().add_widget(widget, position=position)
        # end if
    # end add_widget

    # region PRIVATE

    # Check if there is already a widget in a zone
    def _check_zone(self, position: int):
        """
        Check if there is already a widget in a zone.

        Args:
            position (int): Zone to check
        """
        for widget, kwargs in self.widgets:
            if kwargs.get('position', 0) == position:
                return True
            # end if
        # end for
        return False
    # end _check_zone

    # endregion PRIVATE

# end SplitContainer

