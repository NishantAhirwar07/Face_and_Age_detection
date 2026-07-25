# Age Detection and Face Bounding Box Tool

This repository contains a Python project that detects human faces in an image and estimates the age group of the detected individuals using deep learning models via OpenCV's DNN module.

## Features
* **Face Detection:** Uses a pre-trained TensorFlow model to locate faces within an image.
* **Age Prediction:** Uses a Caffe model to classify detected faces into specific age brackets.
* **Visual Annotation:** Automatically draws bounding boxes around detected faces and labels them with their predicted age ranges.

## Required Pre-trained Weights & Configurations
Before running the script, you need to download and place the following model files in the root directory:

1. **Face Detector Configuration:** `opencv_face_detector.pbtxt`
2. **Face Detector Weights:** `opencv_face_detector_uint8.pb`
3. **Age Predictor Configuration:** `age_deploy.prototxt`
4. **Age Predictor Weights:** `age_net.caffemodel`
