# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Imports
from typing import Tuple, Sequence
import os
import cv2
import numpy as np
import urllib.request as urlreq

from pixelprism.base.image import Image


# Face detection preprocessing
def face_detection_preprocessing(
        image: Image
) -> Image:
    """
    Preprocess the image for face detection

    Args:
        image (Image): Image to preprocess for face detection

    Returns:
        Tuple[Image, np.ndarray]: Preprocessed image and grayscale image
    """
    # set dimension for cropping image
    x, y, width, depth = 50, 200, 950, 500
    image_cropped = image.data[y:(y + depth), x:(x + width)]

    # convert image to Grayscale
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    return Image.from_numpy(image_gray)
# end face_detection_preprocessing


# Face detection
def face_detection(
        image: Image,
        haarcascade: str,
        haarcascade_url: str,
) -> Sequence[Sequence[int]]:
    """
    Detect faces in an image using the Haar Cascade Classifier.

    Args:
        image (Image): Image to detect faces in
        haarcascade (str): Haar Cascade XML file
        haarcascade_url (str): URL to download the Haar Cascade XML file
    """
    # Check if file is in working directory
    if haarcascade not in os.listdir(os.curdir):
        # Download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
    # end if

    # Image preprocessing
    image = face_detection_preprocessing(image)

    # Create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image.data)

    return faces
# end face_detection

