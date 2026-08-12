import cv2
import mediapipe as mp
import time
import math

# --- Initialize MediaPipe's face detection module ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam")
    exit()

# --- FPS tracking (rolling average) ---
previous_time = 0
fps_history = []

# --- Target lock-on state ---
# locked_target stores the last known info about the face we are tracking.
# It starts as None because at launch, no face has been locked yet.
locked_target = None

# How many consecutive frames we tolerate WITHOUT seeing the locked face
# before giving up and re-acquiring a new target (e.g. person briefly
# turned their head, or a hand passed in front of the camera).
MAX_LOST_FRAMES = 15
lost_frames_counter = 0

# Maximum pixel distance allowed between the last known position and a
# new detection for it to still be considered "the same person".
# Prevents accidentally jumping to a completely different face that
# appears far away.
MAX_MATCH_DISTANCE = 150


def get_face_center(detection, frame_width, frame_height):
    """Convert a MediaPipe detection's relative bounding box into
    pixel coordinates, and return the center point + box size."""
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * frame_width)
    y = int(bbox.ymin * frame_height)
    w = int(bbox.width * frame_width)
    h = int(bbox.height * frame_height)
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y, w, h


while True:
    ret, frame = video.read()
    if not ret:
        print("Erreur de lecture de la frame")
        break

    frame_height, frame_width, _ = frame.shape
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Build a simple list of (center_x, center_y, width, height, detection)
    # for every face found this frame, so we can compare them below.
    current_faces = []
    if results.detections:
        for detection in results.detections:
            cx, cy, w, h = get_face_center(detection, frame_width, frame_height)
            current_faces.append((cx, cy, w, h, detection))

    # --- CASE 1: we have no target yet -> lock onto the first face seen ---
    if locked_target is None and current_faces:
        cx, cy, w, h, detection = current_faces[0]
        locked_target = {"center": (cx, cy), "size": (w, h)}
        lost_frames_counter = 0
        print("Cible verrouillée.")

    # --- CASE 2: we already have a target -> find the closest match ---
    elif locked_target is not None and current_faces:
        last_x, last_y = locked_target["center"]

        best_match = None
        best_distance = None

        for cx, cy, w, h, detection in current_faces:
            # Euclidean distance between last known position and this face
            distance = math.hypot(cx - last_x, cy - last_y)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = (cx, cy, w, h, detection)

        # Only accept the match if it's close enough to be plausible
        if best_match is not None and best_distance <= MAX_MATCH_DISTANCE:
            cx, cy, w, h, detection = best_match
            locked_target = {"center": (cx, cy), "size": (w, h)}
            lost_frames_counter = 0
        else:
            # No face was close enough to our last known position this frame
            lost_frames_counter += 1

    # --- CASE 3: we have a target but NO faces were detected at all ---
    elif locked_target is not None and not current_faces:
        lost_frames_counter += 1

    # --- Give up the target if it's been missing too long ---
    if locked_target is not None and lost_frames_counter > MAX_LOST_FRAMES:
        print("Cible perdue, en attente d'un nouveau visage.")
        locked_target = None
        lost_frames_counter = 0

    # --- Draw everything ---
    for cx, cy, w, h, detection in current_faces:
        is_locked = (
            locked_target is not None
            and locked_target["center"] == (cx, cy)
        )
        # Green box for the locked target, gray for any other face seen
        color = (0, 255, 0) if is_locked else (150, 150, 150)
        top_left = (cx - w // 2, cy - h // 2)
        bottom_right = (cx + w // 2, cy + h // 2)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)

    if locked_target is not None:
        lx, ly = locked_target["center"]
        error_x = lx - frame_center_x
        error_y = ly - frame_center_y
        cv2.putText(frame, f"Error X: {error_x}  Error Y: {error_y}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(frame, (lx, ly), 5, (0, 0, 255), -1)
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 255, 0), -1)
    else:
        cv2.putText(frame, "Recherche d'un visage...",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- FPS display ---
    current_time = time.time()
    instant_fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Target Lock - Appuyez sur 'q' pour quitter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()