import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# --- EAR Calculation ---
def eye_aspect_ratio(eye_points, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C), pts

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Eye landmark indices (MediaPipe 468-point model) ---
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# --- Constants ---
EAR_THRESHOLD   = 0.25
FRAME_THRESHOLD = 20

# --- Webcam ---
cap           = cv2.VideoCapture(0)
frame_counter = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            left_EAR,  left_pts  = eye_aspect_ratio(LEFT_EYE,  lm, w, h)
            right_EAR, right_pts = eye_aspect_ratio(RIGHT_EYE, lm, w, h)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Draw eye contours
            cv2.polylines(frame, [np.array(left_pts)],  True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_pts)], True, (0, 255, 0), 1)

            # Drowsiness logic
            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                frame_counter = 0

            # Status display
            status = "DROWSY" if avg_EAR < EAR_THRESHOLD else "ALERT"
            color  = (0, 0, 255) if avg_EAR < EAR_THRESHOLD else (0, 255, 0)

            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()