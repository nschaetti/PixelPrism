#
# Description: A 2D space that can contain other drawables.
#

# Imports
from .base import Drawable


class Space2D(Drawable):

    def __init__(self):
        self.drawables = []
    # end __init__

    def add(self, drawable):
        self.drawables.append(drawable)
    # end add

    def draw(self, context):
        for drawable in self.drawables:
            drawable.draw(context)
        # end for
    # end draw

# end Space2D

