# Fake-Face-Image-Classifier-System README.md
```markdown
# Fake Face Image Classifier System
A binary deep learning classification project to distinguish real human facial photographs from AI-generated fake faces, implemented with custom CNN baseline and MobileNetV2 transfer learning models.

## Project Overview
This system builds two convolutional neural network models to identify visual differences between authentic real faces and synthetic AI fake faces.
1. Lightweight scratch-trained custom CNN as the baseline benchmark
2. Improved MobileNetV2 transfer learning model pre-trained on ImageNet for higher classification accuracy
All codes support dataset splitting, image augmentation, model training, evaluation and result visualisation (confusion matrix, classification metrics).

## Repository File Structure
```
Fake-Face-Image-Classifier-SYSTEM/
├── data_split/               # Split dataset folders (train / val / test)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                    # Saved trained model weights (.keras format)
│   ├── A_CNN_baseline.keras   # Custom scratch CNN baseline model
│   └── B_MobileNet_baseline.keras # Improved MobileNetV2 transfer learning model
├── notebooks/                 # Jupyter notebooks for data preprocessing experiments
│   ├── data_aug.ipynb         # Data augmentation trial notebook
│   └── data_cleaning.ipynb    # Dataset cleaning and quality check notebook
├── src/                       # Core source code scripts
│   ├── check_images.py         # Script to verify image file integrity
│   ├── data_augmentation.py    # Implement random image augmentation pipeline
│   ├── dataset_analysis.py     # Dataset distribution and sample visual analysis
│   ├── evaluation.py           # Model evaluation: confusion matrix, precision/recall/F1/accuracy calculation
│   ├── split_data.py           # Automatically split raw dataset into train/validation/test sets
│   ├── model.py                # Define custom CNN and MobileNetV2 model architectures
│   └── train.py                # Main training script for both baseline and improved model
├── venv312/                    # Local Python 3.12 virtual environment (ignored by git)
├── .gitignore                  # Git ignore file for environment, large datasets and cache
├── README.md                   # Project documentation
└── requirements.txt            # All Python library dependencies
```

## Dataset Notice
The full raw dataset is excluded from this repository due to large file size limits.
This project uses the **Hard Fake vs Real Faces** public dataset from Kaggle, which contains balanced real and AI-generated fake facial samples for binary classification.
Kaggle Dataset Link: https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

### Dataset Folder Setup
1. Download the dataset zip file from Kaggle
2. Extract all images and organise into `data_split/train`, `data_split/val`, `data_split/test` folders
3. Each split subfolder contains two subdirectories: `real` and `fake` for two label classes
Or run `src/split_data.py` to auto-split full raw dataset into standard train/val/test partitions.

## Environment & Setup Guide
### 1. Clone Repository
```bash
git clone https://github.com/pomelocheah/Fake-Face-Image-Classifier-System.git
cd Fake-Face-Image-Classifier-SYSTEM
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment (Python 3.12 recommended)
python -m venv venv312
```
#### Activate Environment
- MacOS / Linux
```bash
source venv312/bin/activate
```
- Windows Command Prompt
```cmd
venv312\Scripts\activate
```

### 3. Install Dependencies
All required libraries are listed in `requirements.txt`
```bash
pip install -r requirements.txt
```

## Core Dependencies
```
Python == 3.12
tensorflow == 2.20.0
Pillow == 12.2.0
scikit-learn == 1.9.0
numpy
matplotlib
```

## Full Project Workflow
1. **Data Preparation & Split**
   Run `src/split_data.py` to separate raw dataset into training, validation and test subsets with fixed random seed for reproducibility.
2. **Data Augmentation**
   Use `src/data_augmentation.py` to add random horizontal flip, rotation, zoom and brightness adjustment on training images to reduce model overfitting.
3. **Model Definition**
   Two model architectures are defined in `src/model.py`:
   - Baseline: Custom lightweight CNN built from scratch without pre-trained weights
   - Improved Model: MobileNetV2 transfer learning with frozen ImageNet pre-trained backbone
4. **Model Training**
   Execute `src/train.py` to train either baseline CNN or MobileNetV2 model; trained weights will auto-save into `/models` folder.
5. **Model Evaluation**
   Run `src/evaluation.py` on saved `.keras` models to output full classification metrics (Accuracy, Precision, Recall, F1-Score) and generate visual confusion matrix plots for test set performance analysis.

## Model Performance Summary
1. Baseline Custom CNN
   - Test Accuracy: 95.88%
   - Weighted Precision: 95.88%
   - Weighted Recall: 95.88%
   - Weighted F1-Score: 95.88%
   Limitation: Shallow layers cannot capture subtle artificial texture defects on fake faces, resulting in minor misclassification.

2. Improved MobileNetV2 Transfer Learning Model
   - Test Accuracy: 98.97%
   - Weighted Precision: 98.99%
   - Weighted Recall: 98.97%
   - Weighted F1-Score: 98.97%
   Advantage: Pre-trained ImageNet weights provide strong universal feature extraction ability, significantly reducing false predictions for both real and fake face categories.

## Author
Multimedia Computing AI System Student, Multimedia University (MMU)
```
```