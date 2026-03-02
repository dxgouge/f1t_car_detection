This repository contains scripts and data configurations for training a YOLO (You Only Look Once) object detection model specifically for car detection using the F1TENTH dataset.
🚀 Overview
The goal of this project is to train a high-performance computer vision model capable of identifying and bounding autonomous racing cars in various environments. The repository is optimized for use with the latest YOLO architectures (e.g., YOLOv11).
📂 Project Structure
f1tenth.v2i.yolov11/: Contains the dataset configuration, including images and labels formatted for YOLO training.
f1tenthRec.py: The main Python script used to initiate and manage the model training process.

🛠️ Requirements
Python: 3.8+
Ultralytics: For YOLO model training and inference.
Hardware: Specify in file if you want CPU or Apple Silicon (M-series) or NVIDIA GPU recommended for acceleration.
📈 Training the Model
To start training the car detection model, run the following command:
bash
python f1tenthRec.py
Use code with caution.

📊 Dataset
The model is trained on the F1TENTH dataset, which features images of small-scale autonomous racing vehicles. This dataset is structured into training, validation, and test sets within the
f1tenth.v2i.yolov11 directory.

# !!Make sure to select an appropriate amount of workers for your hardware and which device to use (CPU/GPU). Batch size 4 with 2 workers on cpu should be more than fine on most systems. 
