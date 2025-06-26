

class DrawableDataMixin:

    def to_drawable(self):
        raise NotImplementedError("Subclasses must implement this method.")
    # end to_drawable

# end DrawableDataMixin
