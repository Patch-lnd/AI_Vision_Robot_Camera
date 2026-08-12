import cv2
import numpy as np
import time

# --- Load OpenCV's built-in face detector (YuNet) and recognizer (SFace) ---
# These are ONNX models downloaded separately (see instructions).
# YuNet: detects face bounding boxes + 5 landmarks (eyes, nose, mouth corners)
# SFace: takes an aligned face crop and outputs a 128-value "fingerprint"
#        (embedding) representing that person's identity
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet.onnx",
    "",
    (320, 320),          # input size the detector expects; we resize internally
    score_threshold=0.7,  # minimum confidence to count as a face
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface.onnx",
    ""
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
locked_embedding = None      # the 128-value "fingerprint" of the tracked person
locked_center = None         # last known pixel position (for drawing only)
last_seen_time = None        # timestamp of the last frame we actually saw them

# How close (in SFace's cosine-similarity score, 0 to 1) a face must be
# to count as "the same person". Higher = stricter match required.
IDENTITY_MATCH_THRESHOLD = 0.36  # SFace's recommended cosine threshold

# How many seconds of continuous absence before we consider the person
# "really really gone" and allow an automatic switch to someone else.
REALLY_LOST_SECONDS = 4.0

previous_time = 0
fps_history = []


def get_embedding(frame, face_box):
    """Align and crop the face using SFace's own helper, then compute
    its embedding vector. face_box is one row from YuNet's output."""
    aligned_face = recognizer.alignCrop(frame, face_box)
    embedding = recognizer.feature(aligned_face)
    return embedding


while True:
    ret, frame = video.read()
    if not ret:
        print("Erreur de lecture de la frame")
        break

    # YuNet expects the frame as-is; it returns an array where each row
    # describes one detected face: [x, y, w, h, landmarks..., score]
    _, faces = detector.detect(frame)

    current_time = time.time()
    best_match_index = None
    best_match_score = -1  # cosine similarity: higher is more similar

    # Store every face's embedding this frame so we can reuse them later
    # (both for identity matching AND for the gesture-triggered switch below)
    # without recomputing anything twice.
    current_embeddings = []

    if faces is not None:
        for i, face_box in enumerate(faces):
            embedding = get_embedding(frame, face_box)
            current_embeddings.append(embedding)

            if locked_embedding is not None:
                # cosine similarity between this face and our locked identity
                score = recognizer.match(
                    locked_embedding, embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > best_match_score:
                    best_match_score = score
                    best_match_index = i

    # --- CASE 1: no target locked yet -> lock onto the first face seen ---
    if locked_embedding is None and faces is not None and len(faces) > 0:
        locked_embedding = get_embedding(frame, faces[0])
        x, y, w, h = faces[0][:4].astype(int)
        locked_center = (x + w // 2, y + h // 2)
        last_seen_time = current_time
        print("Identité verrouillée.")

    # --- CASE 2: we have a target -> did we find a confident match? ---
    elif locked_embedding is not None and best_match_index is not None \
            and best_match_score >= IDENTITY_MATCH_THRESHOLD:
        face_box = faces[best_match_index]
        x, y, w, h = face_box[:4].astype(int)
        locked_center = (x + w // 2, y + h // 2)
        last_seen_time = current_time
        # Slowly refresh the stored embedding so it adapts to lighting/angle
        # changes over time, without fully forgetting the original identity
        new_embedding = get_embedding(frame, face_box)
        locked_embedding = 0.9 * locked_embedding + 0.1 * new_embedding

    # --- CASE 3: target locked but not confidently seen this frame ---
    # We do NOT drop the lock immediately — only after REALLY_LOST_SECONDS
    elif locked_embedding is not None:
        time_since_seen = current_time - last_seen_time
        if time_since_seen > REALLY_LOST_SECONDS:
            print("Personne vraiment perdue depuis", int(time_since_seen), "s -> reverrouillage libre.")
            locked_embedding = None
            locked_center = None

    # --- Draw all detected faces (gray), highlight the locked one (green) ---
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

    # --- FPS (rolling average) ---
    instant_fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Identity Lock - 'q' quitter, 'v' simuler geste", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        # TEMPORARY stand-in for the real "V" gesture, until MediaPipe Hands
        # is wired in. Pressing 'v' means "deliberately switch to the next
        # closest OTHER face" — not just drop the lock and hope for the best.
        print("Geste simule -> recherche de la prochaine cible.")

        candidates = []  # will hold (area, index) for every face that is
                          # NOT the person currently locked on

        if faces is not None:
            for i, face_box in enumerate(faces):
                embedding = current_embeddings[i]

                # If we currently have a locked identity, skip any face
                # that matches it — we don't want to "switch" back onto
                # the same person we're trying to move away from.
                if locked_embedding is not None:
                    score = recognizer.match(
                        locked_embedding, embedding,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )
                    if score >= IDENTITY_MATCH_THRESHOLD:
                        continue  # this is still the same person, skip it

                w, h = face_box[2], face_box[3]
                area = w * h  # bigger box = closer person (our distance proxy)
                candidates.append((area, i))

        if candidates:
            # Sort by area, biggest (closest) first, and take that one
            candidates.sort(reverse=True)
            _, best_index = candidates[0]
            face_box = faces[best_index]

            locked_embedding = current_embeddings[best_index]
            x, y, w, h = face_box[:4].astype(int)
            locked_center = (x + w // 2, y + h // 2)
            last_seen_time = current_time
            print("Nouvelle cible verrouillee.")
        else:
            # No other person available to switch to — stay locked on the
            # current target exactly as-is, don't touch anything. This is
            # the expected behavior with only one face in frame.
            print("Une seule personne visible, on reste sur la cible actuelle.")

video.release()
cv2.destroyAllWindows()