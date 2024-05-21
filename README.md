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

```bash
pip install -r requirements.txt
```
## Usage

### Running the Streamlit App

To start the Streamlit app for real-time machine status monitoring, use:

```bash
streamlit run machine_st_test.py
```
## Script Overview

The machine_st_test.py script loads the trained model and provides a Streamlit interface for making predictions and monitoring machine status.

### Example Usage
1. Ensure you have Streamlit installed.
2. Run the script:

```bash
streamlit run machine_st_test.py
```
3. Follow the prompts in the Streamlit interface to upload a video or use a video stream and monitor the machine status.

## Model Training

To train the model, follow these steps:

1. Ensure your dataset is organized in the following structure:

├── train
│   ├── Moving
│   └── Not Moving
└── test
    ├── class1
    └── class2

