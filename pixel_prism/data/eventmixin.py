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

class EventMixin:
    """
    Abstract class for objects with events.
    """

    def __init__(self):
        """
        Initialize the scalar value.
        """
        super(object).__init__()

        # List of event listeners (per events)
        self.event_listeners = {}
    # end __init__

    def add_event_listener(self, event, listener):
        """
        Add an event listener to the data object.

        Args:
            event (str): Event to listen for
            listener (function): Listener function
        """
        if event not in self.event_listeners:
            raise ValueError(f"Event '{event}' is not supported by this data object.")
        # end if
        self.event_listeners[event].append(listener)
    # end add_event_listener

    def remove_event_listener(self, event, listener):
        """
        Remove an event listener from the data object.

        Args:
            event (str): Event to remove listener from
            listener (function): Listener function to remove
        """
        if event in self.event_listeners:
            self.event_listeners[event].remove(listener)
        # end if
    # end remove_event_listener

    def dispatch_event(self, event, *args, **kwargs):
        """
        Dispatch an event to all listeners.

        Args:
            event (str): Event to dispatch
        """
        if event in self.event_listeners:
            for listener in self.event_listeners[event]:
                listener(*args, **kwargs)
            # end for
        else:
            raise ValueError(f"Event '{event}' is not supported by this data object.")
        # end if
    # end dispatch_event

    # List of events
    def events(self):
        """
        Return the list of events that this data object can dispatch.
        """
        return list(self.event_listeners.keys())
    # end events

# end EventMixin

