# ✍️ Handwritten Digit Recognition using a CNN

This project showcases a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits from the famous MNIST dataset. The model demonstrates a standard and effective approach to image classification tasks.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-blue.svg)

---

## 📖 Project Overview

The primary goal is to build a highly accurate classifier for handwritten digits (0-9). The project covers the full machine learning pipeline: loading and preprocessing the image data, designing a robust CNN architecture, training the model efficiently with callbacks, and evaluating its performance on an unseen test set.

---

## 📊 The MNIST Dataset

The MNIST (Modified National Institute of Standards and Technology) dataset is a cornerstone of the machine learning community and is often referred to as the "Hello, World!" of computer vision. It provides a clean, well-structured benchmark for image classification algorithms.

* **Total Images:** The dataset contains a total of **70,000** grayscale images of handwritten digits.
* **Data Split:** It is pre-divided into two sets:
    * **Training Set:** 60,000 images.
    * **Testing Set:** 10,000 images.
* **Image Properties:**
    * **Dimensions:** Each image is a fixed size of **28x28 pixels**.
    * **Color:** The images are grayscale, meaning they have a single color channel.
* **Classes:** There are **10 classes**, corresponding to the digits 0 through 9. The dataset is well-balanced, with a roughly equal number of images for each digit.

For this project, the initial 60,000 training images were further split, creating a dedicated validation set to monitor the model's performance during training and prevent overfitting.

---

## 🧠 Model Architecture: Convolutional Neural Network (CNN)

A Convolutional Neural Network was chosen for this task due to its exceptional ability to capture spatial hierarchies and patterns in image data. The architecture is built sequentially using Keras:

1.  **Input Layer:** Accepts the 28x28 pixel images with a single channel (`(28, 28, 1)`).
2.  **First Convolutional Block:**
    * A `Conv2D` layer with 32 filters and a `ReLU` activation function to learn basic features like edges and curves.
    * A `MaxPooling2D` layer to downsample the feature maps, making the model more efficient and robust to variations in digit placement.
3.  **Second Convolutional Block:**
    * A `Conv2D` layer with 64 filters and `ReLU` activation to learn more complex patterns from the features identified in the previous block.
    * Another `MaxPooling2D` layer for further downsampling.
4.  **Classifier Head:**
    * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    * A `Dense` layer with 128 neurons and `ReLU` activation.
    * A `Dropout` layer with a rate of 0.5 to randomly deactivate neurons during training, which is a powerful technique to prevent overfitting.
    * The final `Dense` output layer with 10 neurons (one for each digit) and a **`softmax`** activation function to output a probability distribution across the 10 classes.

---

## ✨ Project Workflow

* **Data Loading & Preprocessing:**
    * The MNIST dataset is loaded directly from `tf.keras.datasets`.
    * Pixel values are **normalized** from the range [0, 255] to [0, 1] to aid in faster and more stable training.
    * Labels are **one-hot encoded** to match the `softmax` output layer.
* **Training:**
    * The model is compiled with the `Adam` optimizer, `CategoricalCrossentropy` loss function (standard for multi-class classification), and monitored using `accuracy` and `recall` metrics.
    * An **`EarlyStopping`** callback is used to halt training when the validation loss stops improving, ensuring the model with the best performance is saved.
* **Evaluation:**
    * The final model is evaluated on the 10,000-image test set to measure its real-world performance.
    * The training and validation loss curves are plotted to visualize the learning process.

---

## 📈 Results

The model achieved outstanding performance on the unseen test data, demonstrating its high accuracy and reliability.

| Metric | Test Set Score |
| :--- | :--- |
| **Accuracy** | **99.15%** |
| **Recall** | **99.11%** |
| **Loss** | **0.0262** |



