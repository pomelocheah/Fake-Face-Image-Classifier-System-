

# 📌 Rewritten README.md

```markdown
# Fake Face Image Classification System

A deep learning-based binary image classification system designed to distinguish between real human facial images and AI-generated fake faces using both a custom CNN baseline model and a MobileNetV2 transfer learning approach.

---

## 📌 Project Overview

This project implements and evaluates two convolutional neural network architectures for fake face detection:

- **Baseline Model:** A custom-built CNN trained from scratch
- **Improved Model:** MobileNetV2 with ImageNet pre-trained weights (transfer learning)

The system includes a complete machine learning pipeline covering:
- Dataset preprocessing and splitting
- Data augmentation strategies
- Model training and optimization
- Performance evaluation (confusion matrix + metrics)
- Model comparison analysis

---

## Project Structure

```text
Fake-Face-Image-Classifier-System/
│
├── app.py                         # Gradio web application
├── model.py                       # CNN and MobileNetV2 model definitions
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── data_split/                    # Dataset
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   ├── val/
│   │   ├── fake/
│   │   └── real/
│   └── test/
│       ├── fake/
│       └── real/
│
├── models/                        # Trained models
│   ├── A_CNN_baseline.keras
│   ├── B_MobileNet_baseline.keras
│  
│
├── notebooks/                     # Jupyter notebooks
│   ├── data_aug.ipynb
│   └── data_cleaning.ipynb
│
├── src/                           # Source code
│   ├── check_images.py            # Verify image integrity
│   ├── data_augmentation.py       # Data preprocessing & augmentation
│   ├── dataset_analysis.py        # Dataset visualization and statistics
│   ├── evaluation.py              # Performance evaluation
│   ├── split_data.py              # Dataset splitting
│   ├── validation.py              # Model validation
│   └── train.py                   # Model training
│

```

---

## 📊 Dataset

This project uses the **Hard Fake vs Real Faces Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

### Dataset Preparation

1. Download dataset from Kaggle
2. Organize into the following structure:

```

data_split/
├── train/
├── val/
└── test/
├── real/
└── fake/

````

Alternatively, use `src/split_data.py` to automatically generate the splits.

---

## ⚙️ Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/pomelocheah/Fake-Face-Image-Classifier-System.git
cd Fake-Face-Image-Classifier-System
````

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* macOS/Linux:

```bash
source venv/bin/activate
```

* Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architectures

### 🔹 Baseline Model (Custom CNN)

A lightweight convolutional neural network trained from scratch.

* Designed for feature extraction from facial images
* Uses convolution + pooling layers for spatial feature learning
* Fully connected layers for binary classification

---

### 🔹 Improved Model (MobileNetV2)

A transfer learning model using pretrained ImageNet weights.

* MobileNetV2 backbone (frozen layers)
* Global Average Pooling for feature compression
* Dense classification head for binary output

---

## 🧪 Data Augmentation

Applied only on training data to improve generalization.

### Baseline Augmentation

* Horizontal flip
* Small rotation
* Slight zoom

### Improved Augmentation

* Stronger rotation
* Zoom variation
* Contrast adjustment
* Brightness adjustment

---

## 🚀 Workflow

1. **Dataset Preparation**

   * Split dataset into train/val/test

2. **Data Augmentation**

   * Apply augmentation only on training set

3. **Model Training**

   * Train CNN baseline model
   * Train MobileNetV2 models

4. **Evaluation**

   * Compute accuracy, precision, recall, F1-score
   * Generate confusion matrix

5. **Comparison**

   * Compare baseline vs transfer learning performance

---

## 📈 Model Performance

### 🔹 Baseline CNN

* Accuracy: 95.88%
* Precision: 95.88%
* Recall: 95.88%
* F1-score: 95.88%

**Observation:** Performs well but limited in capturing fine-grained facial artifacts.

---

### 🔹 MobileNetV2 (Improved)

* Accuracy: 98.97%
* Precision: 98.99%
* Recall: 98.97%
* F1-score: 98.97%

**Observation:** Transfer learning significantly improves feature extraction and reduces misclassification.

---

## 📊 Evaluation Output

The system generates:

* Confusion Matrix
* Classification Report (Precision / Recall / F1-score)
* Training Accuracy/Loss Curves

---

## 👨‍💻 Author

Multimedia Computing AI Student
Multimedia University (MMU)

---

## 📌 Notes

* Dataset is not included due to size limitations
* Models are trained locally using TensorFlow 2.20
* Designed for educational and academic research purposes


