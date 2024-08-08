#
# Data classes for Pixel Prism
#


class Data(object):
    """
    A data class that holds data.
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

    def set(self, *args, **kwargs):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        raise NotImplementedError(f"{self.__class__.__name__}.set method must be implemented in subclass.")
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.get method must be implemented in subclass.")
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.copy method must be implemented in subclass.")
    # end copy

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__str__ method must be implemented in subclass.")
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__repr__ method must be implemented in subclass.")
    # end __repr__

# end Scalar
