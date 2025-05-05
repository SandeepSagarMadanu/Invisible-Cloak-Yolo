import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have the YOLOv8 model

# Check if the model file exists
model_path = "cloak_refiner.h5"
if os.path.exists(model_path):
    cloak_refiner = load_model(model_path)  # Load your trained model
    use_dnn = True
    print("Loaded deep learning model for refinement.")
else:
    print("Warning: cloak_refiner.h5 not found! Falling back to traditional masking.")
    use_dnn = False

# Capture video
cap = cv2.VideoCapture(0)
cv2.namedWindow("Invisible Cloak", cv2.WINDOW_NORMAL)

# Allow camera to warm up and capture background frame
print("Capturing background...")
for i in range(60):
    ret, background = cap.read()
background = cv2.flip(background, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    results = model(frame, conf=0.5)
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for cloak (adjust for different cloaks)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    cloak_mask = mask1 + mask2
    cloak_mask = cv2.medianBlur(cloak_mask, 5)
    
    if use_dnn:
        # Process the mask using the deep neural network for improved accuracy
        refined_mask = cloak_refiner.predict(np.expand_dims(cloak_mask, axis=[0, -1]))
        refined_mask = (refined_mask[0, :, :, 0] * 255).astype(np.uint8)
    else:
        refined_mask = cloak_mask  # Use traditional mask if model is missing
    
    # Create inverse mask
    mask_inv = cv2.bitwise_not(refined_mask)
    
    # Use masks to extract background and frame
    bg_part = cv2.bitwise_and(background, background, mask=refined_mask)
    fg_part = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Merge background and foreground
    final_output = cv2.addWeighted(bg_part, 1, fg_part, 1, 0)
    
    # Display results
    cv2.imshow("Invisible Cloak", final_output)
    
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
