

class Layer:
    """
    Class to represent a layer in an image
    """

    def __init__(
            self,
            name,
            image,
            blend_mode='normal',
            active=True
    ):
        """
        Initialize the layer with a name, image, blend mode, and active status

        Args:
            name (str): Name of the layer
            image (np.ndarray): Image data for the layer
            blend_mode (str): Blend mode for the layer
            active (bool): Whether the layer is active
        """
        self.name = name
        self.image = image
        self.blend_mode = blend_mode
        self.active = active
    # end __init__

    def __repr__(self):
        """
        Return a string representation of the layer
        """
        return f"Layer(name={self.name}, blend_mode={self.blend_mode}, active={self.active}, image={self.image})"
    # end __repr__

# end Layer

