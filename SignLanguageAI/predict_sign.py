import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained KNN model
model = pickle.load(open('knn_model.pkl', 'rb'))

# Load the reference ASL chart image
ref_img = cv2.imread("asl_chart.png.jpg")  # Make sure this image exists
ref_img = cv2.resize(ref_img, (400, 400))  # Resize to match webcam frame height

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 400))  # Resize frame to match reference image height

    # Convert the frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Collect 63 features (x, y, z)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(landmarks) == 63:
            prediction = model.predict([landmarks])[0]
            cv2.putText(frame, f'Predicted: {prediction}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Incomplete hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Combine webcam frame and reference chart side by side
    combined_view = cv2.hconcat([frame, ref_img])

    # Show the combined window
    cv2.imshow("Sign Language Detection + Reference", combined_view)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
