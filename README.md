# Classify-gestures-by-reading-muscle-activity (EMG Gesture Classification using Machine Learning)

A comprehensive machine learning project for classifying hand gestures from EMG (Electromyography) signals, designed to support open-source prosthetic control systems.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Project Objectives](#project-objectives)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🎯 Overview

This project implements and compares multiple machine learning models to classify hand gestures from EMG sensor data. The work supports the development of an open-source prosthetic control system that enables prosthetic devices to have multiple degrees of freedom.

The system aims to map user residual muscle gestures to specific actions of a prosthetic, such as opening/closing a hand or rotating a wrist.

**Related Project:** [Cyber Punk Me - Open Source Prosthetic Control](https://github.com/cyber-punk-me)

**Reference Video:** [Living with a mind-controlled robot arm](https://www.youtube.com/watch?v=9NOncx2jU0Q)

---

## 📊 Dataset Description

### Context

The dataset was collected using a **MYO armband** equipped with 8 EMG sensors placed on the skin surface. Each sensor measures electrical activity produced by muscles beneath the skin.

### Data Structure

- **Sensors:** 8 EMG sensors
- **Sampling Rate:** 200 Hz
- **Data Format:** Each row contains 64 columns of EMG data + 1 label column
  ```
  [8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][8sensors][GESTURE_CLASS]
  ```
- **Time Window:** Each row represents 8 consecutive readings = 40ms of recording time (8 × 1/200 seconds)

### Gesture Classes

| Class | Gesture | Description |
|-------|---------|-------------|
| 0 | Rock | Closed fist (as in rock-paper-scissors game) |
| 1 | Scissors | Index and middle finger extended, others closed |
| 2 | Paper | All fingers extended (as in rock-paper-scissors game) |
| 3 | OK | Index finger touching thumb, other fingers spread |

### Recording Details

- **Recording Duration:** 6 recordings × 20 seconds per gesture = 120 seconds total per gesture
- **Total Samples:** 11,678 samples across all 4 gesture classes
- **Recording Method:** Each gesture was prepared and held in a fixed position during recording
- **Data Source:** Recorded from the same right forearm in a short timespan
- **Data Files:** 
  - `0.csv` - Rock gesture
  - `1.csv` - Scissors gesture
  - `2.csv` - Paper gesture
  - `3.csv` - OK gesture

### Dataset Statistics

- **Total Features:** 64 (8 sensors × 8 time steps)
- **Total Samples:** 11,678
- **Class Distribution:** ~2,920 samples per class (balanced dataset)
- **Data Range:** EMG values typically range from -128 to 127

---

## 🎯 Project Objectives

1. **Data Preprocessing:** Load, clean, and prepare EMG signal data for analysis
2. **Feature Engineering:** Standardize features for optimal model performance
3. **Model Training:** Implement and train multiple machine learning classifiers
4. **Model Evaluation:** Compare models using accuracy, confusion matrices, and classification reports
5. **Model Selection:** Identify the best-performing model for gesture classification

---

## 🛠️ Requirements

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/emg-gesture-classification.git
   cd emg-gesture-classification
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place the 4 CSV files (0.csv, 1.csv, 2.csv, 3.csv) in the appropriate input directory
   - Update file paths in the code if needed

---

## 🚀 Usage

### Quick Start

```python
# Run the complete analysis
python emg_classification.py
```

### Step-by-Step Execution

The code is organized into clear sections:

1. **Data Loading:** Combines all 4 gesture CSV files into a single dataset
2. **EDA:** Visualizes data distribution and checks for missing values
3. **Data Preparation:** Splits data into train/test sets and applies feature scaling
4. **Model Training:** Trains 4 different classifiers
5. **Evaluation:** Generates confusion matrices and classification reports
6. **Comparison:** Compares all models and identifies the best performer

---

## 🤖 Models Implemented

### 1. Multi-Layer Perceptron (MLP) Neural Network
- **Architecture:** 2 hidden layers (100, 50 neurons)
- **Activation:** ReLU
- **Optimizer:** Adam
- **Features:** Early stopping with validation
- **Best For:** Complex non-linear patterns

### 2. Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Regularization:** C=1.0
- **Features:** Automatic gamma scaling
- **Best For:** High-dimensional data classification

### 3. Logistic Regression
- **Solver:** LBFGS
- **Type:** Multinomial classification
- **Features:** Fast training, interpretable results
- **Best For:** Baseline linear classification

### 4. Random Forest
- **Trees:** 150 estimators
- **Max Depth:** 12
- **Features:** Ensemble learning with regularization
- **Best For:** Robust classification with feature importance

---

## 📈 Results

### Model Performance Summary

| Model | Train Accuracy | Test Accuracy | Overfitting Gap |
|-------|----------------|---------------|-----------------|
| **MLP Neural Network** | 92.4% | 92.4% | 0.0% |
| **Random Forest** | 93.7% | 91.7% | 2.0% |
| **SVM** | 95.3% | 91.2% | 4.1% |
| **Logistic Regression** | 38.5% | 34.9% | 3.6% |

### Key Findings

✅ **Best Overall Model:** MLP Neural Network
- Highest test accuracy (92.4%)
- Excellent generalization (no overfitting)
- Balanced performance across all gesture classes

✅ **Runner-up:** Random Forest
- Very close performance (91.7% test accuracy)
- Good interpretability through feature importance
- Minimal overfitting

❌ **Poor Performance:** Logistic Regression
- Low accuracy (~35%) indicates non-linear patterns in data
- Linear model insufficient for complex EMG signals

### Per-Class Performance (MLP Model)

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Rock (0) | 0.96 | 0.94 | 0.95 |
| Scissors (1) | 0.93 | 0.93 | 0.93 |
| Paper (2) | 0.91 | 0.91 | 0.91 |
| OK (3) | 0.90 | 0.91 | 0.90 |

---

## 📁 File Structure

```
emg-gesture-classification/
│
├── data/
│   ├── 0.csv                    # Rock gesture data
│   ├── 1.csv                    # Scissors gesture data
│   ├── 2.csv                    # Paper gesture data
│   └── 3.csv                    # OK gesture data
│
├── emg_classification.py        # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── outputs/                     # Generated plots and results
    ├── confusion_matrices/
    └── comparison_plots/
```

---

## 🔬 Technical Details

### Data Preprocessing
- **Missing Values:** None detected in dataset
- **Feature Scaling:** StandardScaler (mean=0, std=1)
- **Train-Test Split:** 80-20 split with stratification
- **Random State:** 42 (for reproducibility)

### Feature Engineering
- **Input Features:** 64 EMG values (8 sensors × 8 time steps)
- **Normalization:** Z-score standardization applied
- **No PCA:** All features retained due to good performance

### Model Selection Criteria
1. Test accuracy (primary metric)
2. Generalization gap (train-test difference)
3. Per-class performance balance
4. Training time and complexity

---

## 🤝 Contributing

Contributions are welcome! This project is part of a larger effort to create open-source prosthetic control systems.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- [ ] Implement real-time classification pipeline
- [ ] Add data augmentation techniques
- [ ] Experiment with deep learning models (CNN, LSTM)
- [ ] Optimize model for embedded deployment
- [ ] Add cross-validation for more robust evaluation
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Create a web demo for model testing

---

## 🙏 Acknowledgments

- **Dataset Source:** [Cyber Punk Me Project](https://github.com/cyber-punk-me)
- **Data Collection App:** [Nukleos](https://github.com/cyber-punk-me/nukleos)
- **Hardware:** MYO Armband (8-channel EMG sensor)
- **Inspiration:** Supporting people who need assistive technology

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Link:** [https://github.com/your-username/emg-gesture-classification](https://github.com/your-username/emg-gesture-classification)
- **Original Project:** [Cyber Punk Me](https://github.com/cyber-punk-me)

---

## 🌟 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{emg_gesture_classification,
  author = {Your Name},
  title = {EMG Gesture Classification using Machine Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/emg-gesture-classification}
}
```

---

## 🚀 Future Work

- [ ] Real-time gesture prediction system
- [ ] Mobile app integration (Android/iOS)
- [ ] Multi-user model adaptation
- [ ] Gesture transition detection
- [ ] Integration with prosthetic control APIs
- [ ] Cross-subject generalization studies
- [ ] Edge device deployment (Raspberry Pi, Arduino)

---

**Let's build technology that helps people! 🦾**

Be one of the real cyber punks inventing electronic appendages and assistive technologies. Together, we can make a real difference in people's lives.
