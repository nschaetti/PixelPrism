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


def call_before(method_name):
    """
    Decorator to call a method before the decorated method is called.

    Args:
        method_name (str): The name of the method to call before the decorated method.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Call the specified method before the actual method
            if hasattr(self, method_name):
                getattr(self, method_name)()
            # end if
            return func(self, *args, **kwargs)
        # end wrapper
        return wrapper
    # end decorator
    return decorator
# end call_before


# Decorator to call a method after another
def call_after(method_name):
    """
    Decorator to call a method after the decorated method is called.

    Args:
        method_name (str): The name of the method to call after the decorated method.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            # Call the specified method after the actual method
            if hasattr(self, method_name):
                getattr(self, method_name)()
            # end if
            return result
        # end wrapper
        return wrapper
    # end decorator
    return decorator
# end call_after

