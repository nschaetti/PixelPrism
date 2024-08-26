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


class Event:
    """
    A basic event.
    """

    def __init__(
            self,
            name: str,
            source: object,
            **kwargs
    ):
        """
        Initialize the event.

        Args:
            name (str): Event name
        """
        self._name = name
        self._source = source
        self._params = kwargs
    # end __init__

    # region PROPERTIES

    @property
    def name(self) -> str:
        return self._name
    # end name

    @property
    def source(self) -> object:
        return self._source
    # end source

    @property
    def params(self) -> dict:
        return self._params
    # end params

    # endregion PROPERTIES

    # region OVERRIDE

    def __getattr__(self, item):
        """
        Called when an attribute is not found in the usual places.
        Looks for the attribute in the params dictionary.
        """
        if item in self._params:
            return self._params[item]
        # end if
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    # end __getattr__

    def __str__(self) -> str:
        return f"Event: {self.name}"
    # end __str__

    # endregion OVERRIDE

# end Event


# Object changed event
class ObjectChangedEvent(Event):
    """
    An object changed event.
    """

    def __init__(
            self,
            source: object,
            **kwargs
    ):
        """
        Initialize the object changed event.
        """
        super().__init__("object_changed", source, **kwargs)
    # end __init__

# end ObjectChangedEvent



