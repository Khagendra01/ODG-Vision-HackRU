import os
import cv2
import numpy as np
import mediapipe as mp

# 1. Import and Install Dependencies
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# 2. Define Functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB to BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Drawing code (same as before)
    pass  # Omitted for brevity; use your existing function here

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in 
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in 
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in 
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 3. Setup Folders for Collection
DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays

# Get list of actions (labels) from the 'train' directory
train_path = 'train'
actions = os.listdir(train_path)
print(f"Actions detected: {actions}")

# Create base directory and action directories if they do not exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

# 4. Collect Keypoint Values for Training and Testing from Videos
mp_holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with mp_holistic_model as holistic:
    # Loop through actions
    for action in actions:
        action_videos_path = os.path.join(train_path, action)
        video_files = [f for f in os.listdir(action_videos_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Processing action '{action}' with {len(video_files)} videos.")

        # Loop through each video file for the current action
        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_videos_path, video_file)

            # Create a directory for each video (sequence)
            sequence_path = os.path.join(DATA_PATH, action, str(video_idx))

            # Check if this video has already been processed
            if os.path.exists(sequence_path):
                # Check if keypoints have been extracted for all frames
                existing_frames = os.listdir(sequence_path)
                if existing_frames:
                    print(f"Skipping video '{video_file}' for action '{action}' (already processed).")
                    continue  # Skip to the next video

            else:
                os.makedirs(sequence_path)

            cap = cv2.VideoCapture(video_path)
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Optional: Draw landmarks on the frame (for visualization)
                # draw_styled_landmarks(image, results)

                # Extract keypoints
                keypoints = extract_keypoints(results)

                # Save keypoints to a .npy file
                npy_path = os.path.join(sequence_path, f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                frame_num += 1

            cap.release()
            print(f"Processed video {video_idx + 1}/{len(video_files)} for action '{action}'.")

print("Data extraction complete.")
