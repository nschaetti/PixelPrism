#
# Description: Module to apply drawing effects to images
#

# Imports
import cv2
import numpy as np

from pixel_prism.base.image import Image


def draw_points(
        image: Image,
        points,
        color,
        thickness
):
    """
    Draw points on an image

    Args:
        image (np.ndarray): Image to draw the points on
        points (list): List of points to draw
        color (tuple): Color of the points
        thickness (int): Thickness of the points
    """
    # Alpha ?
    has_alpha = image.has_alpha

    # Create a temporary image
    temp_image = np.zeros_like(image.data[:, :, :3])

    # Draw the points on the temporary image
    for point in points:
        cv2.circle(
            temp_image,
            (int(point.x), int(point.y)), int(point.size / 2),
            color[:3],
            thickness,
            lineType=cv2.LINE_AA
        )
    # end for

    # Add the points to the image
    mask = temp_image > 0
    image.data[:, :, :3][mask] = temp_image[mask]

    # if has_alpha:
    if has_alpha:
        # print(f"Before: {np.mean(image.data[:, :, 3])}")
        image.data[:, :, 3][mask[:, :, 1]] = 255
        # print(f"After: {np.mean(image.data[:, :, 3])}")
    # end if

    return image
# end draw_points
