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
from typing import Callable, List, Any


# Call a function at a specific time
class Call:
    """
    A class to represent a function call.
    """

    def __init__(
            self,
            func: Callable,
            times: List[float],
            values: List[Any]
    ):
        """
        Initialize the function call.

        Args:
            func (callable): Function to call
            times (list): List of times to call the function
            values (list): List of values to pass to the function
        """
        # Check that "times" and "values" have the same size
        assert len(times) == len(values), "Times and values must have the same size"

        # Members of values must be list of dict
        for value in values:
            assert isinstance(value, dict) or isinstance(value, list), "Values must be a list of dict"
        # end for

        # Properties
        self._func = func
        self._times = times
        self._values = values

        # List of boolean to keep states
        self._states = [False] * len(times)
    # end __init__

    # Update
    def update(
            self,
            t
    ):
        """
        Update the function call.

        Args:
            t (float): Time of the update
        """
        for i, time in enumerate(self._times):
            if t >= time and not self._states[i]:
                # It's a list
                if isinstance(self._values[i], list):
                    self._func(*self._values[i])
                else:
                    self._func(self._values[i])
                # end if
                self._states[i] = True
            # end if
        # end for
    # end update

# end Call
