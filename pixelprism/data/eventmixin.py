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
        # List of event listeners (per events)
        self.event_listeners = {}
    # end __init__

    def add_event(self, event_name):
        """
        Add an event to the data object.

        Args:
            event_name (str): Event to add
        """
        if event_name not in self.event_listeners:
            self.event_listeners[event_name] = []
        # end if
    # end add_event

    def add_event_listener(self, event_name, listener):
        """
        Add an event listener to the data object.

        Args:
            event_name (str): Event to listen for
            listener (function): Listener function
        """
        if event_name not in self.event_listeners:
            raise ValueError(f"Event '{event_name}' is not supported by this data object.")
        # end if

        if listener not in self.event_listeners[event_name]:
            self.event_listeners[event_name].append(listener)
        # end if
    # end add_event_listener

    def remove_event_listener(self, event_name, listener):
        """
        Remove an event listener from the data object.

        Args:
            event_name (str): Event to remove listener from
            listener (function): Listener function to remove
        """
        if event_name in self.event_listeners:
            self.event_listeners[event_name].remove(listener)
        # end if
    # end remove_event_listener

    def dispatch_event(self, event_name, event):
        """
        Dispatch an event to all listeners.

        Args:
            event_name (str): Event to dispatch
            event (object): Event to dispatch
        """
        if event_name in self.event_listeners:
            for listener in self.event_listeners[event_name]:
                listener(event)
            # end for
        else:
            raise ValueError(f"Event '{event_name}' is not supported by this data object.")
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

