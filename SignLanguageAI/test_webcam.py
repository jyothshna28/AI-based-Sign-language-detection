import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

X = []
y = []

# Use DirectShow backend (if not already working, try V4L2 backend)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Alternative: V4L2 (Linux/Mac)

if not cap.isOpened():
    print("❌ Failed to open webcam")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Show your hand sign and press 'a' to save it as label 'A'")
print("Press ESC to stop and train the model")

label = "A"  # You can change this to other letters later

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if key == ord('a'):
                X.append(landmarks)
                y.append(label)
                print("✅ Saved one sample for label:", label)

    cv2.imshow("Sign Language Data Collection", frame)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

if len(X) > 0:
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as knn_model.pkl")
else:
    print("❌ No data captured.")
