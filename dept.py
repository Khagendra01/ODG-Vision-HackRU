import cv2
import numpy as np
import time

# Load YOLOv3-tiny object detection model
net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")
# For faster processing on Raspberry Pi, set preferable backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
classes = []
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up video capture using OpenCV's VideoCapture
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
time.sleep(2.0)  # Camera warm-up time

# Load MiDaS small model for depth estimation
midas = cv2.dnn.readNet("model-small.onnx")

# Set backend and target for MiDaS
midas.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
midas.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Transformation parameters for MiDaS
midas_transform = cv2.dnn.blobFromImage

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame is not captured properly

    frame = cv2.flip(frame, -1)  # Flip if necessary
    h, w = frame.shape[:2]

    # Object Detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # the first five elements are objectness score and bounding box
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                
                # Rectangle coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Depth Estimation
    input_blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), (123.675, 116.28, 103.53), swapRB=True, crop=False)
    midas.setInput(input_blob)
    depth_map = midas.forward()
    depth_map = depth_map[0, :, :]
    depth_map = cv2.resize(depth_map, (w, h))
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, width, height = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            # Estimate distance using depth map
            obj_center_x = int(x + width / 2)
            obj_center_y = int(y + height / 2)
            distance = depth_map[obj_center_y, obj_center_x]
            # Display bounding box and distance
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}, Dist: {distance:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display the frame and depth map
    cv2.imshow("Frame", frame)
    cv2.imshow("Depth Map", depth_map)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
