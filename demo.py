import cv2

from fer.posterv2.face_detector import FaceDetector, NoFaceDetectedException
from fer.posterv2.posterv2_recognizer import PosterV2Recognizer

face_detector = FaceDetector()
facial_expression_recognizer = PosterV2Recognizer()

TEXT_MARGIN = 5
LINE_HEIGHT = 20
FONT_SCALE = 0.5
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_face_bounding_box_and_emotions(frame, face_coordinates, emotions):
    """
    Draws the face bounding box and the emotion probabilities on the frame.
    """
    x1, y1, x2, y2 = face_coordinates

    # Draw the face bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate space needed for text
    text_height = len(emotions) * LINE_HEIGHT + TEXT_MARGIN

    # Check if there's enough space below the face; if not, put the text above
    if y2 + text_height > frame.shape[0]:
        label_position = (x1, y1 - TEXT_MARGIN)
        step = -LINE_HEIGHT
    else:
        label_position = (x1, y2 + LINE_HEIGHT + TEXT_MARGIN)
        step = LINE_HEIGHT

    # Display the emotion probabilities
    for index, emotion in enumerate(facial_expression_recognizer.emotion_labels):
        cv2.putText(frame, f'{emotion}: {emotions[index]:.2f}',
                    (label_position[0], label_position[1] + index * step),
                    FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print('Error: Camera not accessible')
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Failed to capture frame')
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                face_image, box = face_detector.detect_face(rgb_image)
                emotions = facial_expression_recognizer.predict_emotions(face_image)
                draw_face_bounding_box_and_emotions(frame, box, emotions)
            except NoFaceDetectedException:
                pass

            cv2.imshow('Facial Expression Recognition', frame)

            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
