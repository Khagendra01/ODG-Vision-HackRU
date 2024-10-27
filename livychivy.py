import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

# 1. Load the trained model
model = load_model('action.h5')

# 2. Load the actions (labels) you used during training
DATA_PATH = os.path.join('MP_Data')
actions = np.array(os.listdir(DATA_PATH))
print(f"Actions: {actions}")

# 3. Set up Mediapipe
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# 4. Define helper functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
    image.flags.writeable = False
    results = model.process(image)  # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Drawing specifications
    mp_drawing_styles = mp.solutions.drawing_styles

    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )

    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )

    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )

    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    # Extract keypoints from the Mediapipe results
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks and results.pose_landmarks.landmark else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in 
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks and results.face_landmarks.landmark else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in 
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks and results.left_hand_landmarks.landmark else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in 
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks and results.right_hand_landmarks.landmark else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Initialize variables
sequence = []
sentence = []
predictions = []
threshold = 0.5  # Adjust as needed
sequence_length = 10
# Start webcam feed
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            # Prepare input data
            input_data = np.expand_dims(sequence, axis=0)

            # Get prediction
            res = model.predict(input_data)[0]
            predictions.append(res)

            # Keep only recent predictions
            if len(predictions) > 10:
                predictions = predictions[-10:]

            # Calculate average prediction
            avg_res = np.mean(predictions, axis=0)
            if np.max(avg_res) > threshold:
                predicted_action = actions[np.argmax(avg_res)]

                # Update sentence
                if len(sentence) > 0:
                    if predicted_action != sentence[-1]:
                        sentence.append(predicted_action)
                else:
                    sentence.append(predicted_action)

                # Limit sentence length
                if len(sentence) > 5:
                    sentence = sentence[-5:]

        # Display predictions
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show image
        cv2.imshow('OpenCV Feed', image)

        # Break on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
