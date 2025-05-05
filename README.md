# 🧥 Invisible Cloak using YOLOv8 and Deep Learning

This project implements a real-time **Invisible Cloak** system using **YOLOv8** for object detection and optional deep learning refinement to accurately mask and blend the user into the background. Inspired by the “invisibility cloak” concept from Harry Potter, this system hides a person wearing a red cloak by replacing the cloak pixels with background content.

## 🧠 About the Project

Using a webcam, the system detects a red-colored cloak in the video stream and replaces the region with a static background image, creating the illusion of invisibility. It utilizes:

- **YOLOv8** from Ultralytics for object detection
- **OpenCV** for image processing and masking
- **TensorFlow/Keras** to optionally enhance mask accuracy using a custom-trained `cloak_refiner.h5` model

If the trained model is unavailable, traditional color masking is used as a fallback.

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8
- TensorFlow / Keras (for optional mask refinement)
- Webcam or live camera input

## 📦 File Structure

- `Invisible Cloak Yolo.py` - Main Python script
- `cloak_refiner.h5` *(optional)* - Trained model to enhance mask accuracy
- `yolov8n.pt` - Pre-trained YOLOv8 weights file

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/invisible-cloak-yolo.git
   cd invisible-cloak-yolo
   ```

2. Install required libraries:
   ```bash
   pip install opencv-python numpy tensorflow ultralytics
   ```

3. Make sure you have:
   - A webcam connected
   - `yolov8n.pt` model file in your working directory
   - (Optional) `cloak_refiner.h5` model for deep learning refinement

4. Run the script:
   ```bash
   python "Invisible Cloak Yolo.py"
   ```

5. Press `q` to quit the application.

## 🧪 Features

- Real-time cloak detection and background replacement
- YOLOv8-based frame analysis
- Optional DNN mask refinement for cleaner segmentation
- Dynamic cloak detection using HSV color thresholds

## 📌 Notes

- This version uses red cloak detection; you can modify HSV values for other colors.
- YOLOv8 must be installed and `yolov8n.pt` must be present in the same folder.

## 📈 Future Enhancements

- Support for multiple cloak colors or patterns
- Model training pipeline for `cloak_refiner.h5`
- Streamlit or Flask web app integration
- Save output video functionality

## 👨‍💻 Author

- M. Sandeep Sagar
- https://github.com/SandeepSagarMadanu

---
