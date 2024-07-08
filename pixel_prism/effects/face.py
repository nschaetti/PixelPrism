
import dlib
import cv2

from pixel_prism.effects import EffectBase
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
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # end __init__

    def apply(self, image, **kwargs):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_image)

        face_points = []
        for face in faces:
            landmarks = self.predictor(gray_image, face)
            for n in range(0, 68):  # 68 points de repère faciaux
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                face_points.append(FacePoint(x, y))

        return face_points
    # end apply

# end FacePointsEffect
