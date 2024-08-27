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
            self.__dict__['_vars'] = {}
        # end __init__

        def __getattr__(self, item):
            return self._vars[item]
        # end __getattr__

        def __setattr__(self, key, value):
            if key == '_vars':
                self.__dict__[key] = value
            else:
                self._vars[key] = value
            # end if
        # end __setattr__

    # end AnimationRegister

# end AnimableMixin

