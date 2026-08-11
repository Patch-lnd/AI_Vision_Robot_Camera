# Minimal test — only checking if mediapipe imports correctly
import mediapipe as mp

print("Mediapipe version:", mp.__version__)
print("Has 'solutions':", hasattr(mp, "solutions"))