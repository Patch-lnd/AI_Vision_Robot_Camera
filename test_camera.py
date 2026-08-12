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
print("Resolution:", video.get(cv2.CAP_PROP_FRAME_WIDTH), "x", video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS reported by camera:", video.get(cv2.CAP_PROP_FPS))

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
            # Get image dimensions (needed to convert normalized coordinates to pixels,
            # and to find the center of the frame)
            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2

            # We only track ONE face for now (the first one found) — multi-face logic
            # comes later when we add the "switch target" gesture feature
            detection = results.detections[0]

            # MediaPipe gives bounding box coordinates as RELATIVE values (0 to 1),
            # not actual pixels — so we multiply by the frame's real width/height
            bbox = detection.location_data.relative_bounding_box
            face_x_pixels = int(bbox.xmin * frame_width)
            face_y_pixels = int(bbox.ymin * frame_height)
            face_width_pixels = int(bbox.width * frame_width)
            face_height_pixels = int(bbox.height * frame_height)

            # Calculate the CENTER of the detected face (not its top-left corner)
            face_center_x = face_x_pixels + (face_width_pixels // 2)
            face_center_y = face_y_pixels + (face_height_pixels // 2)

            # This is the key value: how far off-center is the face?
            # Positive error_x = face is to the RIGHT of center -> arm should rotate right
            # Negative error_x = face is to the LEFT of center -> arm should rotate left
            # Same logic vertically for error_y (tilt up/down)
            error_x = face_center_x - frame_center_x
            error_y = face_center_y - frame_center_y

            # Display these values on screen so we can watch them change live
            cv2.putText(frame, f"Error X: {error_x}  Error Y: {error_y}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw a small dot at the face's center, and one at the frame's center,
            # so you can visually see the offset we just calculated
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 255, 0), -1)
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
