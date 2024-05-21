# Monitoring-Vision-AI

This repository provides a comprehensive solution for monitoring machine status using a TensorFlow/Keras model. It features a Streamlit web interface for real-time predictions, motion detection, status logging, and email alerts. The system captures video, preprocesses frames, and determines machine status for efficient management.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#Script-Overview)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to develop a machine learning model to detect and monitor the status of machines using a convolutional neural network (CNN). The model leverages TensorFlow and Keras for building and training the network, and Streamlit for creating an interactive user interface for predictions.

## Features

- **Data Preprocessing and Augmentation:** Automated data handling and augmentation for better model training.
- **Convolutional Neural Network Architecture:** Efficient and effective model structure for accurate predictions.
- **Model Training with TensorFlow and Keras:** End-to-end training pipeline.
- **Real-time Predictions with Streamlit:** Interactive web interface for real-time monitoring.
- **Motion Detection:** Enhanced monitoring with integrated motion detection.
- **Status Logging:** Comprehensive logging of machine status over time.
- **Email Alerts:** Immediate email notifications for status changes.

## Installation

To install the required dependencies, run:

```python <- to be performed in Terminal/Command Prompt
pip install -r requirements.txt
```
## Usage

### Running the Streamlit App

To start the Streamlit app for real-time machine status monitoring, use:

```python
streamlit run machine_st_test.py
```
## Script Overview

The machine_st_test.py script loads the trained model and provides a Streamlit interface for making predictions and monitoring machine status.

### Example Usage
1. Ensure you have Streamlit installed.
2. Run the script:

```python
streamlit run machine_st_test.py
```
3. Follow the prompts in the Streamlit interface to upload a video or use a video stream and monitor the machine status.

## Model Training

To train the model, follow these steps:

1. Ensure your dataset is organized in the following structure:
```bash
├── train
│   ├── Moving
│   └── Not Moving
└── test
    ├── Moving
    └── Not Moving
```
2. Update the `train_data_dir` and `test_data_dir` variables in the training script to point to your dataset directories.
3. Run the model training script:
4. The trained model will be saved to the specified path.

![modeltraining](https://github.com/debarun1234/Monitoring-Vision-AI/assets/17366387/43b48e03-5c84-4d0a-bfc8-6aeb68b7fbc4)

## Model Deployment
### Streamlit App Deployment

1. Load the saved model in the Streamlit app:
   ```python
   model = tf.keras.models.load_model('machine_model_5jan10pm.h5')
   ```
2. Update the video_path variable to point to your video stream or file:
   ```python
   video_path = "rtsp://admin:Admin@123@125.19.34.95:554/cam/realmonitor?channel=1&subtype=0"
   ```
3. Run the Streamlit app to monitor the machine status:
   ```python
   streamlit run machine_st_test.py
   ```

![application_deployment_sequence](https://github.com/debarun1234/Monitoring-Vision-AI/assets/17366387/1b0446be-b346-4126-93b8-4625e9eaaaea)



## Evaluation

Evaluate the model's performance using the provided evaluation scripts. Metrics such as accuracy, precision, and recall are used to measure performance.

### Evaluation Metrics

- **Accuracy:** Measure of correct predictions.
- **Precision:** Measure of the accuracy of the positive predictions.
- **Recall:** Measure of the ability to detect positive instances.

## Results

The model achieved an accuracy of X% on the test dataset. Below are some sample results:

![modelsummary](https://github.com/debarun1234/Monitoring-Vision-AI/assets/17366387/3602e878-5e9f-4be1-91c8-b608e853ded3)


## Contributing

I welcome contributions! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GPL License v3 - see the LICENSE.md file for details.
