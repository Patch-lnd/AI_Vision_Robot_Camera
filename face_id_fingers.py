import cv2
import numpy as np
import mediapipe as mp
import time

# --- Load OpenCV's built-in face detector (YuNet) and recognizer (SFace) ---
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet.onnx",
    "",
    (320, 320),
    score_threshold=0.7,
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface.onnx",
    ""
)

# --- Load MediaPipe Hands for gesture detection ---
# model_complexity=0 -> lightest/fastest model, plenty accurate for one
# simple gesture like this.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,  # accepts either hand, or both at once
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((frame_width, frame_height))

frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# --- Identity lock-on state ---
locked_embedding = None
locked_center = None
last_seen_time = None

IDENTITY_MATCH_THRESHOLD = 0.36
REALLY_LOST_SECONDS = 7.0

# --- Gesture stability state ---
# We require the V sign to be held for several consecutive frames before
# triggering a switch, and we enforce a cooldown afterwards so holding the
# hand up doesn't cause repeated switches.
GESTURE_HOLD_FRAMES_REQUIRED = 10
gesture_hold_counter = 0
GESTURE_COOLDOWN_SECONDS = 2.0
last_switch_time = 0

# How far (in multiples of the tracked face's width) a hand is allowed to
# be from the tracked person's face and still count as "their hand".
# Scales naturally with distance: a farther person has a smaller face in
# pixels, and correspondingly a smaller pixel "arm reach" too.
HAND_REACH_MULTIPLIER = 3.5

previous_time = 0
fps_history = []


def get_embedding(frame, face_box):
    """Align and crop the face using SFace's own helper, then compute
    its embedding vector. face_box is one row from YuNet's output."""
    aligned_face = recognizer.alignCrop(frame, face_box)
    embedding = recognizer.feature(aligned_face)
    return embedding


def is_v_sign(hand_landmarks):
    """Check whether a detected hand is making a 'V' (peace sign):
    index and middle finger extended, ring and pinky folded down.
    Landmark y values are smaller near the top of the image, so an
    'extended' finger has its tip ABOVE (smaller y than) its middle
    joint (the PIP joint)."""
    lm = hand_landmarks.landmark

    index_extended = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_extended = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_folded = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > \
        lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_folded = lm[mp_hands.HandLandmark.PINKY_TIP].y > \
        lm[mp_hands.HandLandmark.PINKY_PIP].y

    return index_extended and middle_extended and ring_folded and pinky_folded


def switch_to_next_target(faces, current_embeddings, locked_embedding):
    """Shared logic for both the real gesture and the keyboard test key.
    Picks the closest OTHER face (excluding whoever is currently locked)
    and returns the new state. If no alternative exists, returns the
    original locked_embedding unchanged (stay on current target)."""
    candidates = []

    if faces is not None:
        for i, face_box in enumerate(faces):
            embedding = current_embeddings[i]

            if locked_embedding is not None:
                score = recognizer.match(
                    locked_embedding, embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score >= IDENTITY_MATCH_THRESHOLD:
                    continue  # still the same person, skip

            w, h = face_box[2], face_box[3]
            area = w * h
            candidates.append((area, i))

    if candidates:
        candidates.sort(reverse=True)
        _, best_index = candidates[0]
        face_box = faces[best_index]
        new_embedding = current_embeddings[best_index]
        x, y, w, h = face_box[:4].astype(int)
        new_center = (x + w // 2, y + h // 2)
        return new_embedding, new_center, True

    # No alternative face — keep current target unchanged
    return locked_embedding, None, False


while True:
    ret, frame = video.read()
    if not ret:
        print("Erreur de lecture de la frame")
        break

    current_time = time.time()

    # ===================== FACE IDENTITY TRACKING =====================
    _, faces = detector.detect(frame)

    best_match_index = None
    best_match_score = -1
    current_embeddings = []
    locked_face_index_this_frame = None  # which face IS the tracked person, this frame

    if faces is not None:
        for i, face_box in enumerate(faces):
            embedding = get_embedding(frame, face_box)
            current_embeddings.append(embedding)

            if locked_embedding is not None:
                score = recognizer.match(
                    locked_embedding, embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > best_match_score:
                    best_match_score = score
                    best_match_index = i

    if locked_embedding is None and faces is not None and len(faces) > 0:
        locked_embedding = get_embedding(frame, faces[0])
        x, y, w, h = faces[0][:4].astype(int)
        locked_center = (x + w // 2, y + h // 2)
        last_seen_time = current_time
        locked_face_index_this_frame = 0
        print("Identité verrouillée.")

    elif locked_embedding is not None and best_match_index is not None \
            and best_match_score >= IDENTITY_MATCH_THRESHOLD:
        face_box = faces[best_match_index]
        x, y, w, h = face_box[:4].astype(int)
        locked_center = (x + w // 2, y + h // 2)
        last_seen_time = current_time
        new_embedding = get_embedding(frame, face_box)
        locked_embedding = 0.9 * locked_embedding + 0.1 * new_embedding
        locked_face_index_this_frame = best_match_index

    elif locked_embedding is not None:
        time_since_seen = current_time - last_seen_time
        if time_since_seen > REALLY_LOST_SECONDS:
            print("Personne vraiment perdue -> reverrouillage libre.")
            locked_embedding = None
            locked_center = None

    # ===================== HAND GESTURE DETECTION =====================
    # MediaPipe expects RGB; OpenCV gives BGR by default.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands_detector.process(rgb_frame)

    v_sign_detected_this_frame = False

    # We can only tell WHOSE hand it is if we actually see the locked
    # person's face this exact frame. If they're momentarily out of view,
    # we simply ignore any hands this frame rather than risk reacting to
    # a bystander's gesture.
    if hand_results.multi_hand_landmarks and locked_face_index_this_frame is not None:
        locked_box = faces[locked_face_index_this_frame]
        lx, ly, lw, lh = locked_box[:4].astype(int)
        locked_face_center_px = (lx + lw // 2, ly + lh // 2)
        max_reach_pixels = HAND_REACH_MULTIPLIER * lw

        # Precompute the pixel centers of every OTHER detected face, so we
        # can check "is this hand closer to MY target than to anyone else"
        other_face_centers = []
        if faces is not None:
            for i, face_box in enumerate(faces):
                if i == locked_face_index_this_frame:
                    continue
                ox, oy, ow, oh = face_box[:4].astype(int)
                other_face_centers.append((ox + ow // 2, oy + oh // 2))

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Use the wrist landmark as the hand's reference position
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_px = (int(wrist.x * frame_width), int(wrist.y * frame_height))

            distance_to_target = np.hypot(
                wrist_px[0] - locked_face_center_px[0],
                wrist_px[1] - locked_face_center_px[1],
            )

            distance_to_others = min(
                (np.hypot(wrist_px[0] - c[0], wrist_px[1] - c[1])
                 for c in other_face_centers),
                default=float("inf")
            )

            belongs_to_target = (
                distance_to_target <= max_reach_pixels
                and distance_to_target < distance_to_others
            )

            # Visual feedback: green wrist dot = counted, red = ignored
            dot_color = (0, 255, 0) if belongs_to_target else (0, 0, 255)
            cv2.circle(frame, wrist_px, 8, dot_color, -1)

            if belongs_to_target and is_v_sign(hand_landmarks):
                v_sign_detected_this_frame = True

    elif hand_results.multi_hand_landmarks:
        # We see hands but not the locked target's face this frame —
        # draw them for visibility, but never let them trigger a switch.
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Track how many CONSECUTIVE frames the V sign has been held
    if v_sign_detected_this_frame:
        gesture_hold_counter += 1
    else:
        gesture_hold_counter = 0

    # Only trigger if held long enough AND we're past the cooldown from
    # the last switch (prevents rapid repeated switching)
    gesture_ready = gesture_hold_counter >= GESTURE_HOLD_FRAMES_REQUIRED
    cooldown_elapsed = (current_time - last_switch_time) >= GESTURE_COOLDOWN_SECONDS

    if gesture_ready and cooldown_elapsed and locked_embedding is not None:
        new_embedding, new_center, switched = switch_to_next_target(
            faces, current_embeddings, locked_embedding
        )
        locked_embedding = new_embedding
        if switched:
            locked_center = new_center
            last_seen_time = current_time
            print("Geste V detecte -> nouvelle cible verrouillee.")
        else:
            print("Geste V detecte, mais une seule personne visible.")
        last_switch_time = current_time
        gesture_hold_counter = 0  # reset so we need a fresh hold next time

    # ===================== DRAWING =====================
    if faces is not None:
        for i, face_box in enumerate(faces):
            x, y, w, h = face_box[:4].astype(int)
            is_locked = (i == best_match_index and locked_embedding is not None)
            color = (0, 255, 0) if is_locked else (150, 150, 150)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if locked_center is not None:
        error_x = locked_center[0] - frame_center_x
        error_y = locked_center[1] - frame_center_y
        cv2.putText(frame, f"Error X: {error_x}  Error Y: {error_y}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif locked_embedding is not None:
        seconds_lost = current_time - last_seen_time
        cv2.putText(frame, f"Recherche... perdu depuis {seconds_lost:.1f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Aucune cible verrouillee",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show gesture hold progress, useful for tuning the required duration
    cv2.putText(frame, f"V hold: {gesture_hold_counter}/{GESTURE_HOLD_FRAMES_REQUIRED}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    instant_fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Identity Lock + Geste V - 'q' quitter, 'v' simuler", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        # Keyboard fallback, kept for quick debugging without needing
        # to actually show your hand to the camera.
        print("Geste simule (clavier).")
        new_embedding, new_center, switched = switch_to_next_target(
            faces, current_embeddings, locked_embedding
        )
        locked_embedding = new_embedding
        if switched:
            locked_center = new_center
            last_seen_time = current_time
        last_switch_time = current_time

video.release()
cv2.destroyAllWindows()
hands_detector.close()