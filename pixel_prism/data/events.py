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
import weakref
from enum import Enum, auto


# Type of events
class EventType(Enum):
    """
    Type of events
    """
    POSITION_CHANGED = auto()
    SCALE_CHANGED = auto()
    ROTATION_CHANGED = auto()
    VALUE_CHANGED = auto()
    FILL_COLOR_CHANGED = auto()
    LINE_COLOR_CHANGED = auto()
    LINE_WIDTH_CHANGED = auto()
    LINE_CAP_CHANGED = auto()
    LINE_JOIN_CHANGED = auto()
    LINE_DASH_CHANGED = auto()
# end EventType


# Event class
class Event:
    """
    Represents an event.
    """

    # Constructor
    def __init__(self):
        self._subscribers = weakref.WeakSet()
    # end __init__

    # Subscribe
    def subscribe(self, callback):
        """
        Subscribe to the event with a weak reference.

        Args:
            callback: Callback function
        """
        self._subscribers.add(callback)
    # end subscribe

    def unsubscribe(self, callback):
        """
        Unsubscribe from the event.

        Args:
            callback: Callback function
        """
        self._subscribers.discard(callback)
    # end unsubscribe

    def trigger(self, *args, **kwargs):
        """
        Trigger the event, calling all subscribers.
        """
        for subscriber in self._subscribers:
            subscriber(*args, **kwargs)
        # end for
    # end trigger

    def listen(self):
        """
        Decorator to subscribe a method to the event.
        """
        def decorator(func):
            self.subscribe(func)
            return func
        # end decorator
        return decorator
    # end listen

    def __iadd__(self, callback):
        """
        Allows using `+=` to add a subscriber.
        """
        if callback:
            self.subscribe(callback)
        # end if
        return self
    # end __iadd__

    def __isub__(self, callback):
        """
        Allows using `-=` to remove a subscriber.
        """
        if callback:
            self.unsubscribe(callback)
        # end if
        return self
    # end __isub__

# end Event



