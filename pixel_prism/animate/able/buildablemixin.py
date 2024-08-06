#
# Description: Mixin class for buildable objects.
#

# Imports
from typing import Any
from .animablemixin import AnimableMixin


class BuildableMixin(AnimableMixin):
    """
    Interface class for buildable objects
    """

    # Initialize
    def __init__(
            self,
            is_built: bool = False,
            build_ratio: float = 0.0,
    ):
        """
        Initialize the buildable object.

        Args:
            is_built (bool): Is the object built
            build_ratio (float): Build ratio
        """
        super().__init__()
        self.is_built = is_built
        self.build_ratio = 1.0 if is_built else build_ratio
    # end __init__

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the object.
        """
        self.build_ratio = 0.0
        self.is_built = False
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the object.
        """
        self.build_ratio = 1.0
        self.is_built = True
    # end end_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        self.build_ratio = interpolated_t
    # end animate_build

# end BuildableMixin
