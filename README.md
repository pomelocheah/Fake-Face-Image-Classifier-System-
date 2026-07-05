# Fake Face Image Classification System

A deep learning-based binary image classification system for distinguishing between **real human faces** and **AI-generated fake faces** using a custom CNN baseline model and a MobileNetV2 transfer learning model.

---

# Project Title

**Fake Face Image Classification System**

---

# Group Members

| Name | Student ID |
|------|------------|
| Cheah Xiao You| 252UA254PY |

---

# Dataset Source

**Hard Fake vs Real Faces Dataset (Kaggle)**

https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

Download the dataset and organize it as follows:

```text
data_split/
├── train/
│   ├── fake/
│   └── real/
├── val/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

---

# Setup Instructions

Clone the repository:

```bash
git clone https://github.com/pomelocheah/Fake-Face-Image-Classifier-System.git
cd Fake-Face-Image-Classifier-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# How to Train the Model

Run:

```bash
python src/train.py
```

The trained models will be saved in the **models/** folder.

---

# How to Evaluate the Model

Run:

```bash
python src/evaluation.py
```

The evaluation script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# How to Run Prediction App

Run:

```bash
python app.py
```

After launching, open the Gradio URL shown in the terminal (for example: http://127.0.0.1:7860).

Upload an image to predict whether it is **Real** or **Fake**.

---

# Example Prediction Output

```
Prediction : Fake Face
Confidence : 98.97%
```

or

```
Prediction : Real Face
Confidence : 99.21%
```

---

# Best Result Summary

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| CNN Baseline | 54.12% | 29.29% | 54.12% | 38.00% |
| MobileNetV2 | 98.97% | 98.97% | 98.97% | 98.97% |

---

# Known Limitations

- The model was trained using a single public dataset.
- Performance may decrease on unseen datasets with different image distributions.
- Only binary classification (Real vs. Fake) is supported.
- The prediction confidence should not be treated as definitive proof of image authenticity.

---

# References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.

2. TensorFlow Documentation  
https://www.tensorflow.org/

3. Keras Documentation  
https://keras.io/

4. Hard Fake vs Real Faces Dataset (Kaggle)  
https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces