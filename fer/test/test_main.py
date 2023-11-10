from pathlib import Path
from unittest import TestCase

from fastapi.testclient import TestClient

from fer.main import app


class MainTest(TestCase):
    client = TestClient(app)

    def test_ping(self):
        # Act
        response = self.client.get('/ping')

        # Assert
        self.assertEquals(200, response.status_code)
        self.assertEquals('ping', response.json())

    def test_recognize_emotions_from_base64_string(self):
        # Arrange
        base64_image = (Path(__file__).parent / 'test_image_base64.txt').read_text('utf-8')

        # Act
        response = self.client.post('/recognize/base64', data=base64_image, headers={'Content-Type': 'text/plain'})

        # Assert
        self.assertEquals(200, response.status_code)
        self.assertEquals(
            ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'],
            list(response.json().keys())
        )

    def test_recognize_emotions_from_file(self):
        # Arrange
        file_name = 'test_image.png'
        image_bytes = (Path(__file__).parent / file_name).read_bytes()

        # Act
        response = self.client.post('/recognize/file', files={'file': ('test_image.png', image_bytes, 'image/png')})

        # Assert
        self.assertEquals(200, response.status_code)
        self.assertEquals(
            ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'],
            list(response.json().keys())
        )
