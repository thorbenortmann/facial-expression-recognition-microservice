import base64
import random
from typing import Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fer.posterv2.face_detector import FaceDetector, NoFaceDetectedException
from fer.posterv2.posterv2_recognizer import PosterV2Recognizer

face_detector = FaceDetector()

emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
facial_expression_recognizer1 = PosterV2Recognizer('11-07-11-59-model_best_state_dict_only.pth', emotion_labels)
facial_expression_recognizer2 = PosterV2Recognizer('11-10-09-22-model_best_state_dict_only.pth', emotion_labels)
facial_expression_recognizers = [facial_expression_recognizer1, facial_expression_recognizer2]

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/ping')
def ping() -> str:
    return 'ping'


@app.post('/recognize/base64')
async def recognize_emotions_from_base64_string(base64_image: str = Body(...)) -> Dict[str, float]:
    image_bytes = base64.b64decode(base64_image)
    return recognize_emotions(image_bytes)


@app.post('/recognize/file')
async def recognize_emotions_from_file(file: UploadFile = File(...)) -> Dict[str, float]:
    image_bytes = await file.read()
    return recognize_emotions(image_bytes)


def recognize_emotions(image_bytes: bytes) -> Dict[str, float]:
    image_byte_array = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(image_byte_array, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    try:
        face_image, _ = face_detector.detect_face(rgb_image)
    except NoFaceDetectedException:
        raise HTTPException(status_code=422, detail="No face detected")

    facial_expression_recognizer = random.choice(facial_expression_recognizers)
    probabilities = facial_expression_recognizer.predict_emotions(face_image)
    return {k: round(v, 4) for k, v in zip(facial_expression_recognizer.emotion_labels, probabilities)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
