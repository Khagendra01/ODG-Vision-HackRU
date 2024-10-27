import cv2
import numpy as np
import os
import pyttsx3
import speech_recognition as sr
import time
import mediapipe as mp
# Initialize TTS engine
engine = pyttsx3.init()

# Initialize speech recognition
recognizer = sr.Recognizer()

# Load face recognizer and face detector
recognizer_face = cv2.face.LBPHFaceRecognizer_create()
recognizer_face.read('trainer/trainer.yml')

# Load both frontal and profile face Haar cascades
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Load names from names.txt
names = {}
if os.path.exists('names.txt'):
    with open('names.txt', 'r') as f:
        for line in f:
            id, name = line.strip().split(',')
            names[int(id)] = name

# Load YOLOv3 model
yolo_path = 'yolo-coco'  # Folder containing YOLO files
labelsPath = os.path.sep.join([yolo_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Generate random colors for labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

# Load the YOLO network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get output layer names
ln = net.getUnconnectedOutLayersNames()



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define ASL alphabet
asl_alphabet = {
    (0, 1, 1, 1, 1): 'A', (1, 0, 0, 0, 0): 'B', (1, 1, 0, 0, 0): 'C',
    (1, 0, 0, 0, 1): 'D', (0, 0, 0, 0, 0): 'E', (1, 1, 1, 0, 0): 'F',
    (0, 1, 1, 1, 0): 'G', (0, 1, 1, 0, 0): 'H', (0, 0, 0, 0, 1): 'I',
    (0, 0, 0, 1, 1): 'J', (1, 1, 0, 0, 1): 'K', (1, 0, 0, 0, 0): 'L',
    (0, 1, 1, 1, 1): 'M', (0, 1, 1, 1, 0): 'N', (1, 1, 1, 1, 1): 'O',
    (1, 1, 1, 0, 0): 'P', (0, 1, 0, 0, 1): 'Q', (1, 1, 0, 0, 0): 'R',
    (1, 1, 1, 1, 0): 'S', (0, 0, 0, 0, 1): 'T', (1, 1, 0, 0, 1): 'U',
    (1, 1, 0, 0, 1): 'V', (1, 1, 1, 0, 1): 'W', (0, 1, 1, 0, 0): 'X',
    (1, 0, 0, 1, 1): 'Y', (1, 0, 0, 0, 1): 'Z'
}

def recognize_asl_gesture(hand_landmarks):
    # Extract fingertip states (1 for open, 0 for closed)
    finger_states = [
        1 if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y else 0,
        1 if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y else 0,
        1 if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y else 0,
        1 if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y else 0,
        1 if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y else 0
    ]
    
    # Convert finger states to tuple for dictionary lookup
    finger_tuple = tuple(finger_states)
    
    # Look up the corresponding ASL letter
    return asl_alphabet.get(finger_tuple, "Unknown")

# Function to ask user for traffic light status
def ask_for_status():
    with sr.Microphone() as source:
        print("Listening for user response...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            # Recognize speech using Google's Speech Recognition API
            user_response = recognizer.recognize_google(audio)
            print(f"User said: {user_response}")
            return user_response.lower()  # Convert to lowercase for easier comparison
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

# Initialize video capture
cam = cv2.VideoCapture(0)
detected_faces = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    H, W = frame.shape[:2]

    # YOLOv3 object detection
    # Create a blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # Initialize lists
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Filter out weak predictions
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate top-left corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non-Maxima Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw bounding box and label
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}%".format(LABELS[classIDs[i]],
                                        confidences[i] * 100)
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if the detected object is a traffic light
            if LABELS[classIDs[i]] == 'traffic light':
                # Capture the image around the traffic light
                roi = frame[y:y + h, x:x + w]

                # Save the ROI image to a file
                if not os.path.exists('traffic_lights'):
                    os.makedirs('traffic_lights')
                img_name = f"traffic_lights/traffic_light_{x}_{y}.png"
                cv2.imwrite(img_name, roi)
                print(f"Traffic light image saved: {img_name}")

                # Use text-to-speech to ask the user
                engine.say("Traffic light detected. Do you want to know the status?")
                engine.runAndWait()

                # Listen for user's response
                user_response = ask_for_status()

                # Check the user's response
                if user_response == "yes":
                    print("User requested traffic light status.")

                    # Optional: Analyze the traffic light status locally
                    # Convert ROI to HSV color space
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Define color ranges for red, yellow, and green
                    red_lower1 = np.array([0, 100, 100])
                    red_upper1 = np.array([10, 255, 255])
                    red_lower2 = np.array([160, 100, 100])
                    red_upper2 = np.array([179, 255, 255])

                    yellow_lower = np.array([15, 100, 100])
                    yellow_upper = np.array([35, 255, 255])

                    green_lower = np.array([40, 100, 100])
                    green_upper = np.array([90, 255, 255])

                    # Create masks for colors
                    mask_red1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
                    mask_red2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
                    mask_red = cv2.add(mask_red1, mask_red2)
                    mask_yellow = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
                    mask_green = cv2.inRange(hsv_roi, green_lower, green_upper)

                    # Count the number of pixels for each color
                    red_pixels = cv2.countNonZero(mask_red)
                    yellow_pixels = cv2.countNonZero(mask_yellow)
                    green_pixels = cv2.countNonZero(mask_green)

                    # Determine the traffic light status
                    if red_pixels > yellow_pixels and red_pixels > green_pixels:
                        status = 'Red'
                    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                        status = 'Yellow'
                    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                        status = 'Green'
                    else:
                        status = 'Unknown'

                    # Use TTS to announce the status
                    engine.say(f"The traffic light is {status}")
                    engine.runAndWait()

                else:
                    print("User did not request traffic light status.")

    # Face recognition
    def detect_and_recognize_faces(frame, names, W, H):
        global detected_faces
        current_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        frontal_faces = frontal_face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H))
        )
        
        # Detect left profile faces
        left_profile_faces = profile_face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H))
        )
        
        # Flip the image horizontally to detect right profile faces
        flipped_gray = cv2.flip(gray, 1)
        right_profile_faces = profile_face_cascade.detectMultiScale(
            flipped_gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H))
        )
        
        # Store the best face data
        best_face = None
        best_confidence = 100  # Lower is better since it's the distance to the match
        best_label = "Unknown"
        
        # Process all types of faces
        for faces, is_flipped in [(frontal_faces, False), (left_profile_faces, False), (right_profile_faces, True)]:
            for (x, y, w, h) in faces:
                if is_flipped:
                    x = W - x - w
                
                label, confidence = recognize_face(gray, x, y, w, h, recognizer_face, names)
                
                if label != "Unknown" and confidence < best_confidence:
                    best_confidence = confidence
                    best_face = (x, y, w, h)
                    best_label = label
        
        # If a best face was found, draw it on the frame and process it
        if best_face:
            x, y, w, h = best_face
            draw_label(frame, x, y, w, h, best_label, best_confidence)
            
            # Check if this face is newly detected
            if best_label not in detected_faces:
                detected_faces[best_label] = current_time
                engine.say(f"   {best_label} is detected.")
                engine.runAndWait()
        
        # Remove faces that haven't been seen for more than 2 minutes
        faces_to_remove = [name for name, last_seen in detected_faces.items() if current_time - last_seen > 120]
        for name in faces_to_remove:
            del detected_faces[name]

    def recognize_face(gray, x, y, w, h, recognizer_face, names):
        # Predict the face
        id, confidence = recognizer_face.predict(gray[y:y + h, x:x + w])
        
        # Confidence check
        if confidence < 100:
            name = names.get(id, "Unknown")
        else:
            name = "Unknown"
        
        return name, confidence

    def draw_label(frame, x, y, w, h, name, confidence):
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        confidence_text = f" {round(100 - confidence)}%"
        cv2.putText(frame, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # Call the detect_and_recognize_faces function
    detect_and_recognize_faces(frame, names, W, H)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize ASL gesture
            asl_letter = recognize_asl_gesture(hand_landmarks)
            
            # Display recognized letter
            cv2.putText(frame, f"ASL: {asl_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face and Object Detection', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
