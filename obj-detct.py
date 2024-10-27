import os
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import time
import mediapipe as mp
from ultralytics import YOLO
from picamera2 import Picamera2
from pymongo import MongoClient
from bson.binary import Binary

# MongoDB setup
client = MongoClient("mongodb://your_mongo_db_uri")
db = client["face_traffic_monitor"]
users_collection = db["users"]
traffic_log_collection = db["traffic_logs"]
face_log_collection = db["face_logs"]

# Initialize TTS engine and speech recognition
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Load face recognizer and face detector
recognizer_face = cv2.face.LBPHFaceRecognizer_create()
recognizer_face.read('trainer/trainer.yml')

# Load face detection cascades
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Function to fetch user names from the database
def load_user_names():
    names = {}
    for user in users_collection.find():
        names[user["_id"]] = user["name"]
    return names

# Function to ask for traffic light status
def ask_for_status():
    with sr.Microphone() as source:
        print("Listening for user response...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            user_response = recognizer.recognize_google(audio)
            print(f"User said: {user_response}")
            return user_response.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

names = load_user_names()
detected_faces = {}

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    H, W = frame.shape[:2]

    # YOLOv8 object detection
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Traffic light detection and logging
        if label == "traffic light":
            roi = frame[y1:y2, x1:x2]
            engine.say("Traffic light detected. Do you want to know the status?")
            engine.runAndWait()

            user_response = ask_for_status()
            if user_response == "yes":
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

                # Define color ranges
                red_lower1, red_upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
                red_lower2, red_upper2 = np.array([160, 100, 100]), np.array([179, 255, 255])
                yellow_lower, yellow_upper = np.array([15, 100, 100]), np.array([35, 255, 255])
                green_lower, green_upper = np.array([40, 100, 100]), np.array([90, 255, 255])

                mask_red = cv2.add(cv2.inRange(hsv_roi, red_lower1, red_upper1), cv2.inRange(hsv_roi, red_lower2, red_upper2))
                mask_yellow = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
                mask_green = cv2.inRange(hsv_roi, green_lower, green_upper)

                red_pixels, yellow_pixels, green_pixels = cv2.countNonZero(mask_red), cv2.countNonZero(mask_yellow), cv2.countNonZero(mask_green)

                # Determine status
                if red_pixels > max(yellow_pixels, green_pixels):
                    status = 'Red'
                elif yellow_pixels > max(red_pixels, green_pixels):
                    status = 'Yellow'
                elif green_pixels > max(red_pixels, yellow_pixels):
                    status = 'Green'
                else:
                    status = 'Unknown'

                # Log to database
                traffic_log_collection.insert_one({"status": status, "timestamp": time.time()})
                engine.say(f"The traffic light is {status}")
                engine.runAndWait()

    # Face recognition
    def detect_and_recognize_faces(frame, names, W, H):
        global detected_faces
        current_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H)))
        left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H)))
        flipped_gray = cv2.flip(gray, 1)
        right_profile_faces = profile_face_cascade.detectMultiScale(flipped_gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * W), int(0.1 * H)))
        
        best_face, best_confidence, best_label = None, 100, "Unknown"
        
        for faces, is_flipped in [(frontal_faces, False), (left_profile_faces, False), (right_profile_faces, True)]:
            for (x, y, w, h) in faces:
                if is_flipped:
                    x = W - x - w
                
                id, confidence = recognizer_face.predict(gray[y:y + h, x:x + w])
                name = names.get(id, "Unknown") if confidence < 100 else "Unknown"
                
                if name != "Unknown" and confidence < best_confidence:
                    best_confidence, best_face, best_label = confidence, (x, y, w, h), name

        if best_face:
            x, y, w, h = best_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{best_label} {100 - int(best_confidence)}%", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if best_label not in detected_faces:
                detected_faces[best_label] = current_time
                face_log_collection.insert_one({"name": best_label, "timestamp": current_time})
                engine.say(f"{best_label} detected.")
                engine.runAndWait()

        # Clear outdated detections
        for name in [name for name, last_seen in detected_faces.items() if current_time - last_seen > 120]:
            del detected_faces[name]

    detect_and_recognize_faces(frame, names, W, H)

    cv2.imshow('Face and Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
