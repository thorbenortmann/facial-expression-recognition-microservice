# Copyright 2022 Andrey Savchenko
# https://github.com/HSE-asavchenko/hsemotion-onnx/blob/bd8a9882924b38e859ae7801305ae203cd71acc1/demo/recognize_emotions_video.py
# Copyright for modifications 2023 Thorben Ortmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
from typing import Tuple

import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh


class NoFaceDetectedException(Exception):
    pass


class FaceDetector:

    def __init__(self):
        self.face_mesh = \
            FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2, min_tracking_confidence=0.2)

    def detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detects a face on the given image and crops it accordingly.
        :param image: numpy array of shape (height, width, channels)
        :return: cropped numpy array of shape (height, width, channels)
        :raise: NoFaceDetectedException if no face is detected.
        """
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise NoFaceDetectedException()

        face_landmarks = results.multi_face_landmarks[0]
        height, width, _ = image.shape
        x1 = y1 = 1
        x2 = y2 = 0
        for landmark in face_landmarks.landmark:
            if landmark.x < x1:
                x1 = landmark.x
            if landmark.y < y1:
                y1 = landmark.y
            if landmark.x > x2:
                x2 = landmark.x
            if landmark.y > y2:
                y2 = landmark.y
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        return image[y1:y2, x1:x2, :], (x1, y1, x2, y2)
