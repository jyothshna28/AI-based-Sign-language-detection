import cv2
import mediapipe as mp
import numpy as np
import os

label = input("Enter the letter (A-Z) to record: ").upper()
save_path = f"data/{label}.npy"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []

print("Press 'c' to capture a sample, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        wrist_x = hand.landmark[0].x
        wrist_y = hand.landmark[0].y

        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])  # normalize to wrist

        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Letter: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and result.multi_hand_landmarks:
        print(f"Captured sample for {label}")
        data.append(landmarks)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save samples
data = np.array(data)
if os.path.exists(save_path):
    existing = np.load(save_path)
    data = np.concatenate((existing, data))

np.save(save_path, data)
print(f"Saved {len(data)} samples for letter '{label}'")

