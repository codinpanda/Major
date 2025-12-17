# Project Report: AI-Based Real-Time Anomaly Detection for Preventive Healthcare

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Literature Survey](#2-literature-survey)
3.  [Software Requirements](#3-software-requirements)
4.  [System Design](#4-system-design)
5.  [Implementation Method](#5-implementation-method)
6.  [Testing](#6-testing)
7.  [Conclusion](#7-conclusion)
8.  [References](#8-references)
9.  [Table of Figures](#table-of-figures)

---

## 1. Introduction

### 1.1 Overview
In modern healthcare, continuous monitoring of physiological signals is crucial for early detection of health anomalies such as cardiac stress, arrhythmia, and autonomic nervous system dysregulation. This project, **Antigravity Health**, is an AI-powered real-time health monitoring system designed to bridge the gap between wearable sensor data and actionable medical insights.

### 1.2 Problem Statement
Traditional wearable devices often provide raw metrics (heart rate, step count) without deep contextual analysis. Users typically receive retrospective reports rather than real-time alerts for critical physiological deviations. This latency in anomaly detection can delay necessary preventative interventions.

### 1.3 Objective
The primary objective is to develop a robust, full-stack application that:
*   Ingests real-time physiological data (ECG, EDA, Heart Rate) from wearable devices (specifically Samsung Galaxy Watch).
*   Processes this data using a lightweight, browser-based Deep Learning model (Hybrid LSTM-GRU).
*   Visualizes health metrics in an intuitive, premium dashboard inspired by Samsung Health.
*   Provides immediate alerts for detected anomalies (stress events, irregular patterns).

---

## 2. Literature Survey

The development of this system builds upon existing research in affective computing and biomedical signal processing.

1.  **WESAD Dataset (Schmidt et al., 2018)**: The *Wearable Stress and Affect Detection* dataset serves as the foundational ground truth for our model. It provides multimodal sensor data (ECG, EDA, EMG) labeled for baseline, stress, and amusement states.
2.  **Deep Learning in Time-Series**: Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), have proven superior to traditional statistical methods for analyzing sequential physiological data due to their ability to retain long-term dependencies.
3.  **Edge AI & ONNX**: Recent advancements in WebAssembly (Wasm) and ONNX Runtime enable sophisticated machine learning models to run directly in the client browser, mitigating privacy concerns associated with cloud-based processing and reducing latency.

---

## 3. Software Requirements

### 3.1 Frontend
*   **Framework**: React 18.3 (TypeScript)
*   **Build Tool**: Vite 5.4
*   **Styling**: Tailwind CSS 3.4 (Utility-first)
*   **State Management**: React Context API
*   **Visualization**: Chart.js 4.4, Recharts (Sparklines)

### 3.2 Backend / AI & Data
*   **Model Training**: Python 3.8, PyTorch
*   **Dataset**: WESAD (ECG, EDA, EMG)
*   **Inference Engine**: ONNX Runtime Web (Wasm backend)
*   **Data Format**: JSON (WebSocket streams), .onnx (Model artifacts)

### 3.3 Mobile & Wearable
*   **Wear OS App**: Kotlin, Android SDK API 30+ (Samsung Galaxy Watch 4/5/6)
*   **Mobile Bridge**: Android App (Kotlin), OkHttp (WebSocket)

### 3.4 Development Tools
*   **IDE**: VS Code, Android Studio
*   **Version Control**: Git, GitHub
*   **Deployment**: Vercel (CI/CD)

---

## 4. System Design

### 4.1 Architecture Diagram
*(Figure 1: High-Level System Architecture)*
> **[Sensor Layer: Samsung Watch]** $\to$ **[Mobile Bridge Layer: Android App]** $\to$ **[Transport Layer: WebSocket]** $\to$ **[Application Layer: React Web App]** $\to$ **[Inference Layer: ONNX Model]**

### 4.2 Data Flow
1.  **Acquisition**: The Wear OS app captures raw sensor events (Heart Rate, Accelerometer) at 1-10Hz.
2.  **Transmission**: Data is broadcast via Bluetooth Low Energy (BLE) / Data Layer API to the paired Android phone.
3.  **Bridging**: The Android 'Mobile' app forwards these packets via WebSocket to the React web application.
4.  **Inference**: The `InferenceEngine` in the browser buffers the data, normalizes it, and passes it through the Hybrid LSTM-GRU model.
5.  **Visualization**: The UI updates continuously with the latest inference probability (Stress Score) and physiological metrics.

### 4.3 Modules
*   **DataGenerator**: Simulates WESAD data or consumes external WebSocket streams.
*   **InferenceEngine**: Manages the ONNX session, handles input tensors, and outputs anomaly probabilities.
*   **GuidanceEngine**: Provides breathing exercises and wellness tips based on the user's current state.
*   **Dashboard UI**: A modular component system (VitalsCard, RealTimeChart) for displaying data.

---

## 5. Implementation Method

### 5.1 Model Training (Python)
We implemented a **Hybrid LSTM-GRU** network. The architecture consists of:
*   Input Layer: Handling multi-channel sensor data (ECG, EDA).
*   LSTM Layer (32 units): Captures long-term temporal dependencies.
*   GRU Layer (16 units): Efficiently processes short-term fluctuations.
*   Dense Output Layer: Sigmoid activation for binary classification (Normal vs. Anomaly).
*   *Optimization*: Used Adam optimizer and Binary Cross-Entropy loss, with **Early Stopping** to prevent overfitting.

### 5.2 Frontend Development (React)
The web application was built using a component-based architecture.
*   **Z-Score Replacement**: The initial statistical anomaly detection was replaced with the `onnxruntime-web` inference engine.
*   **Performance**: Utilized `requestAnimationFrame` for smooth 60fps chart rendering.
*   **Responsiveness**: Implemented a "Mobile-First" design using Tailwind CSS grid and flexbox utilities.

### 5.3 Wearable Integration
A custom Android project (`AntigravityWear`) was developed.
*   **Wear OS Module**: Uses `SensorManager` to access hardware sensors.
*   **Data Layer**: Implements Google's Play Services Wearable API for reliable watch-to-phone communication.

---

## 6. Testing

### 6.1 Unit Testing
*   **Model Validation**: The Python model achieved **~89% accuracy** on the test split of the WESAD dataset.
*   **Robustness**: Implemented Gaussian Noise Injection ($\sigma=0.1$) during training to ensure the model remains accurate even with noisy real-world sensor data.

### 6.2 Integration Testing
*   **Data Pipeline**: Verified end-to-end data flow from the `mock_mobile_bridge.js` script to the frontend WebSocket client.
*   **Latency**: Measured system latency from signal generation to visualization at <100ms on local networks.

### 6.3 User Acceptance Testing (UAT)
*   **Visual Verification**: Confirmed that charts update smoothly without jitter.
*   **Alert Accuracy**: Verified that simulated stress events trigger the visual "High Stress" warnings in the dashboard.
*   **Deployment**: Validated successful production build and deployment on Vercel (resolved 404 errors by correct root directory configuration).

---

## 7. Conclusion

The **Antigravity Health** system successfully demonstrates the feasibility of real-time, edge-based anomaly detection using consumer wearables. By shifting the inference workload to the client browser (ONNX) and leveraging deep learning (LSTM-GRU), we achieved a system that is both private and responsive. The integration of the WESAD dataset ensures scientifically grounded analysis, while the Samsung Health-inspired UI provides a professional user experience. Future work will focus on expanding the wearable support to other platforms (Apple Watch, Garmin) and incorporating cloud-based long-term trend analysis.

---

## 8. References
1.  Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. ICMI.
2.  Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
3.  ONNX Runtime Documentation. *https://onnxruntime.ai/*
4.  React Documentation. *https://react.dev/*
5.  Google Wear OS Developer Guide. *https://developer.android.com/training/wearables*

---

## Table of Figures
*   **Figure 1**: High-Level System Architecture Diagram (See Section 4.1)
*   **Figure 2**: Hybrid LSTM-GRU Model Architecture (See Section 5.1)
*   **Figure 3**: Frontend Dashboard UI - Live Monitor (See Section 5.2)
*   **Figure 4**: Anomaly Detection Alert Flow (See Section 4.2)
