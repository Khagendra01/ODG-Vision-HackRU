import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
# Load pre-trained model (assuming a CNN-LSTM model trained on gesture sequences)
model = load_model('asl_word_recognition_model.h5')

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process a sequence of hand gestures and predict words
def recognize_asl_word(sequence_of_hand_landmarks):
    # Convert keypoints to input format (flatten or normalize)
    input_sequence = np.array(sequence_of_hand_landmarks).reshape(1, -1)  # Assuming flattening keypoints

    # Predict the word
    predicted_word = model.predict(input_sequence)
    
    return predicted_word

# Capture video feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints and append to sequence
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append((landmark.x, landmark.y, landmark.z))
            
            # Predict the word when a complete gesture sequence is formed
            predicted_word = recognize_asl_word(keypoints)
            print(f"Predicted Word: {predicted_word}")
    
    # Display the frame
    cv2.imshow('ASL Word Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
