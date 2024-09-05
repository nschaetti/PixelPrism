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

#
# Description: This file contains the Able class, which is a subclass of the Drawable class.
#

class AnimableMixin:
    """
    Abstract class for animatable objects.
    """

    class AnimationRegister:
        """
        Animation register.
        """

        def __init__(self):
            """
            Constructor
            """
            self.__dict__['_vars'] = {}
        # end __init__

        def __getattr__(self, item):
            """
            Get item
            """
            return self._vars[item]
        # end __getattr__

        def __setattr__(self, key, value):
            """
            Set item
            """
            if key == '_vars':
                self.__dict__[key] = value
            else:
                self._vars[key] = value
            # end if
        # end __setattr__

    # end AnimationRegister

    # Animable registry
    class AnimableRegistry:
        """
        Keep a list of animations for the object
        """

        def __init__(self):
            self._animators = list()
            self._index = 0
        # end if

        @property
        def animators(self):
            return self._animators
        # end animators

        # Add
        def add(self, animator):
            self._animators.append(animator)
        # end add

        # Delete
        def remove(self, animator):
            self._animators.remove(animator)
        # end remove

        def __iter__(self):
            self._index = 0  # Reset index at the start of iteration
            return self
        # end __iter__

        def __next__(self):
            if self._index < len(self._animators):
                item = self._animators[self._index]
                self._index += 1
                return item
            else:
                raise StopIteration  # Required to end iteration
            # end if
        # end __next__

    # end AnimableRegistry


    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # List of animations
        self._animable_registry = self.AnimableRegistry()
    # end __init__

    @property
    def animable_registry(self):
        return self._animable_registry
    # end animable_registry

# end AnimableMixin

