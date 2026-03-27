import cv2
import mediapipe as mp
import pickle
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

label = input("Enter the label (A-Z): ")

if os.path.exists("sign_data.pkl"):
    with open("sign_data.pkl", "rb") as f:
        data, labels = pickle.load(f)

cap = cv2.VideoCapture(0)
print("Press 's' to save hand landmarks. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            key = cv2.waitKey(1)
            if key == ord('s'):
                data.append(landmark_list)
                labels.append(label)
                print(f"Saved sample {len(data)}")

    cv2.imshow("Capture Hand Sign", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open("sign_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print("Data saved to sign_data.pkl")