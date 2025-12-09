# Visual Wake Words (VWW) for Edge AI

## 📌 Project Overview
**Visual Wake Words (VWW)** represents a class of tiny, low-power computer vision models designed to detect the presence of a specific object (usually a person) to "wake up" a larger system. This project implements an end-to-end pipeline to train, quantize, and evaluate deep learning models suitable for deployment on resource-constrained Microcontrollers (MCUs) like the ESP32 or Arduino Nano.

**Goal:** Create a model that fits within **<512KB SRAM** and **<1MB Flash** while maintaining high detection accuracy.

## 🛠️ Running Environment
This project is optimized for execution in a cloud environment with GPU acceleration.

*   **Platform:** [Google Colab](https://colab.research.google.com/) (Recommended)
*   **Hardware Accelerator:** **T4 GPU**
*   **Language:** Python 3.x
*   **Framework:** TensorFlow 2.x / Keras

> **Note:** While the code can run on a local machine with a CPU, training will be significantly slower. Using a T4 GPU in Colab is highly recommended for faster experimentation.

## 📂 Dataset
We use the **INRIA Person Dataset**, a benchmark dataset for pedestrian detection.
*   **Classes:** Person (Positive) vs. Background (Negative).
*   **Preprocessing:** Images are resized to **96x96** pixels.
*   **Augmentation:** Random flips and brightness adjustments are applied to improve robustness.

## 🧠 Models Evaluated
We compare three candidate architectures designed for efficiency:

1.  **MobileNetV2 (alpha=0.35):** A highly efficient architecture using Inverted Residuals.
2.  **MobileNetV3-Small (Minimalistic):** Optimized for mobile devices (uses ReLU instead of HardSwish for better MCU compatibility).
3.  **SimpleCNN:** A custom 3-layer baseline CNN to demonstrate the need for modern architectures.

## 📉 Key Features & Workflow
1.  **Data Engineering:** Automatic download, extraction, and restructuring of the INRIA dataset.
2.  **Exploratory Data Analysis (EDA):** Visualization of class distribution and image dimensions.
3.  **Model Training:** Training with `EarlyStopping` and `ModelCheckpoint`.
4.  **Quantization:** Converting models from **Float32** to **Int8 TFLite** format for 4x size reduction and faster inference on MCUs.
5.  **Comprehensive Evaluation:**
    *   **F1-Score:** To handle class imbalance.
    *   **Model Size (Flash):** Ensuring it fits on the chip.
    *   **Peak RAM (SRAM):** Estimating memory usage for activations.
    *   **MACs:** Proxy for latency and power consumption.

## 🏆 Results Summary
| Model | Type | F1 Score | Size (KB) | Peak RAM (KB) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | **Int8** | **High** | **~600** | **Low** | **✅ Selected** |
| MobileNetV3 | Int8 | Medium | ~500 | Low | Good Alternative |
| SimpleCNN | Int8 | Low | ~100 | **Very High** | Not Recommended |

**MobileNetV2** was selected as the optimal model for its balance of high accuracy and low resource consumption.

## 🚀 How to Run
1.  Upload the notebook `vww_complete_project2.ipynb` to Google Colab.
2.  Go to **Runtime** > **Change runtime type**.
3.  Select **T4 GPU** as the hardware accelerator.
4.  Run all cells sequentially.

## 📚 References
*   [Visual Wake Words Dataset](https://arxiv.org/abs/1906.05721)
*   [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
*   [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
