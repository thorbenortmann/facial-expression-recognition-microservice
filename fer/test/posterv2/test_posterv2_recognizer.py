from unittest import TestCase

from fer.posterv2.face_detector import FaceDetector
from fer.posterv2.posterv2_recognizer import PosterV2Recognizer
from fer.test.utils import load_file_to_cv2_rgb_array


class PosterV2RecognizerTest(TestCase):

    def test_predict_emotions(self):
        # Arrange
        rgb_image = load_file_to_cv2_rgb_array('test_image.png')
        cropped_image, _ = FaceDetector().detect_face(rgb_image)
        emotion_recognizer = PosterV2Recognizer()

        # Act
        recognized_emotions = emotion_recognizer.predict_emotions(cropped_image)

        # Assert
        self.assertTrue(len(recognized_emotions) == 7)
        for re in recognized_emotions:
            self.assertIsInstance(re, float)
