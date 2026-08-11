import cv2
import mediapipe as mp 
import time

#Initialize Mediapipe's face detection module 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# model_selection=0 -> optimized for faces within ~2 meters (webcam range)
# model_selection=1 -> optimized for faces further away (up to ~5 meters)
# min_detection_confidence -> minimum confidence score (0-1) to count as a detected face

face_detection = mp_face_detection.FaceDetection(
    model_selection = 0,
    min_detection_confidence=0.6
)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Erreur :  Impossibel d'ouvrir la webcam")
    exit()

# VARIABLES for FPS calculations
previous_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Erreur de lecture de al frame")
        break

    # MediaPipe expects RGB images, but OpenCV captures in BGR by default
    # so we convert the color format before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Run face detection on the current frame
    # This returns a 'results' object containing all detected faces

    results = face_detection.process(rgb_frame)

    # If at least one face was detected, results.detections will not be None
    if results.detections:
        for detection in results.detections:
            #Draw the bounding box and confidence score directly on the frame 
            mp_drawing.draw_detection(frame, detection)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1/(current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Face Detection 'q' pour quitter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()





















""" import cv2

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Ereur : Impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("Erreur de lecture de la frame")
        break

    cv2.imshow("Webcam - Appuyez sur 'q' pour quitter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

 """
