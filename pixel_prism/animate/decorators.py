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


# Attribute decorator
def animeattr(attr_name):
    """
    Decorator to mark a specific attribute for inspection within an animeclass.

    Args:
        attr_name (str): The name of the attribute to inspect.
    """
    def decorator(cls):
        if not hasattr(cls, '_attrs_to_inspect'):
            cls._attrs_to_inspect = []
        # end if

        # Add attributes
        cls._attrs_to_inspect.append(attr_name)

        # Add methods if not already present
        if not hasattr(cls, 'animeclass_attributes'):
            # Public method to get the list of inspectable attributes
            def animeclass_attributes(cls):
                return cls._attrs_to_inspect
            # end animeclass_attributes
            cls.animeclass_attributes = classmethod(animeclass_attributes)
        # end if

        # Add animation method
        if not hasattr(cls, 'is_animeclass'):
            # Add a public method to check if the class is inspectable
            def is_animeclass(cls):
                return getattr(cls, '_inspect_for_propagation', False)
            # end is_animeclass
            cls.is_animeclass = classmethod(is_animeclass)
        # end if

        return cls
    # end decorator

    return decorator
# end animeattr

