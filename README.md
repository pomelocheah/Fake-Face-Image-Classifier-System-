# Fake-Face-Image-Classifier-System



# Fake vs Real Image Classification Project

## 📌 Project Overview
This project is a machine learning / deep learning classification system designed to distinguish between **fake and real images**.

The dataset contains two categories:
- Real images
- Fake images

A model is trained to learn patterns and perform binary classification.

---

## 📁 Repository Structure

```

fake-face-classifier/
├── src/                  # Source code (training & prediction scripts)
├── data/                # Dataset (NOT included in repository)
│   ├── real/
│   └── fake/
├── requirements.txt    # Python dependencies
├── README.md

```

---

## 🚫 Dataset Notice

The dataset is not included in this repository due to size limitations.

To run the project, please manually prepare the dataset:

```

data/
├── real/
└── fake/

````

---

## ⚙️ Setup Instructions (For Lecturer / Reviewer)

Follow these steps to run the project:

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pomelocheah/Fake-Face-Image-Classifier-System.git
cd fake-face-classifier
````

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Mac / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

This will automatically install all required libraries for the project.

---

## 🚀 How to Run the Project



## 🧠 Workflow Summary

1. Load dataset (fake vs real images)
2. Preprocess images
3. Train classification model
4. Evaluate performance
5. Run predictions on new images

---

## 📦 Requirements

Main dependencies include:

* Python 3.x
* NumPy
* OpenCV
* TensorFlow / PyTorch
* scikit-learn
* Matplotlib

(Full list in `requirements.txt`)

---

## 👨‍💻 Author

AI System Students in MMU




