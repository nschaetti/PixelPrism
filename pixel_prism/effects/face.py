

import cv2

from pixel_prism.effects import EffectBase, DrawPointsEffect
from pixel_prism.primitives import Point


class FacePoint(Point):
    def __init__(self, x, y, size=1):
        super().__init__(x, y, size)

    def __repr__(self):
        return f"FacePoint(x={self.x}, y={self.y}, size={self.size})"
    # end FacePoint

# end FacePoint


class FacePointsEffect(EffectBase):
    """
    Face points effect to detect facial landmarks on an image
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.landmark_model = cv2.face.createFacemarkLBF()
        self.landmark_model.loadModel(cv2.data.haarcascades + "lbfmodel.yaml")
    # end __init__

    def apply(
            self,
            image_obj,
            input_layers,
            output_layers,
            **kwargs
    ):
        """
        Apply the face points effect to the image

        Args:
            image_obj (Image): Image object to apply the effect to
            input_layers (list): List of input layer names
            output_layers (list): List of output layer names
            kwargs: Additional keyword arguments
        """
        for layer_name in input_layers:
            layer = image_obj.get_layer(layer_name)
            if layer:
                image = layer.image
                gray_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray_image,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                face_points = []
                if len(faces) > 0:
                    _, landmarks = self.landmark_model.fit(gray_image, faces)
                    for landmark_set in landmarks:
                        for point in landmark_set[0]:
                            face_points.append(FacePoint(point[0], point[1]))
                        # end for
                    # end for
                # end if

                for output_layer_name in output_layers:
                    output_layer = image_obj.get_layer(output_layer_name)
                    if output_layer:
                        output_layer.image = DrawPointsEffect(face_points).apply(output_layer.image)
                    # end if
                # end for
            # end if
        # end for
    # end apply

# end FacePointsEffect
