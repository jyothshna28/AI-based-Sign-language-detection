# AI-based-Sign-language-detection
I developed an AI-based Sign Language Detection system that recognizes hand gestures corresponding to English alphabet letters (A–Z) using a webcam in real time.
The core objective was to bridge the communication gap between hearing-impaired individuals and those who do not understand sign language. I used MediaPipe to extract 3D hand landmarks, trained a K-Nearest Neighbors (KNN) model on those features using Python, and built a prediction interface to classify the signs as text. The system also displays a reference chart and overlays the predicted letter on the screen.

I collected and trained the model on my own hand gesture dataset to ensure accuracy and flexibility, making it a foundational step toward building a complete word or sentence recognition system in the future.
