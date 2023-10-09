import base64
from typing import Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fer.hsemotion.face_detector import FaceDetector, NoFaceDetectedException
from fer.hsemotion.hsemotion_recognizer import HSEmotionRecognizer

face_detector = FaceDetector()
facial_expression_recognizer = HSEmotionRecognizer()
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
        face_image = face_detector.detect_face(rgb_image)
    except NoFaceDetectedException:
        raise HTTPException(status_code=422, detail="No face detected")

    probabilities = facial_expression_recognizer.predict_emotions(face_image)
    return {
        'anger': round(probabilities[0], 4),
        'disgust': round(probabilities[1], 4),
        'fear': round(probabilities[2], 4),
        'happiness': round(probabilities[3], 4),
        'neutral': round(probabilities[4], 4),
        'sadness': round(probabilities[5], 4),
        'surprise': round(probabilities[6], 4),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
