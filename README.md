# Crop Classification & Clustering System

A machine learning-based system that predicts the most suitable crop based on environmental conditions and analyzes agricultural patterns using clustering techniques.

---

# Features

* Predict best crop using environmental inputs
* Analyze agricultural patterns using KMeans clustering
* Compare multiple machine learning models
* Dimensionality reduction using PCA
* Save and reuse trained models (.pkl files)
* Feature engineering for improved accuracy

---

# Machine Learning Approach

## Supervised Learning (Crop Prediction)

Models used:

* Logistic Regression
* Random Forest (Best Performing)
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Ensemble (Voting Classifier)

---

## Unsupervised Learning (Clustering)

* KMeans Clustering
* Optimal clusters selected using Silhouette Score
* PCA applied before clustering to improve performance

---

# Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

# Dataset

* Source: ICRISAT District-Level Agricultural Dataset
* Features include:

  * Rainfall
  * Temperature
  * Soil moisture
  * Fertilizer usage
  * Crop yield
  * Nutrient values (N, P, K)

---

# Project Workflow

```
Data Collection
      ↓
EDA (Exploratory Data Analysis)
      ↓
Data Cleaning & Outlier Handling
      ↓
Feature Engineering
      ↓
Feature Scaling (StandardScaler)
      ↓
Model Training (Multiple Models)
      ↓
Model Evaluation (Accuracy, F1 Score)
      ↓
PCA (Dimensionality Reduction)
      ↓
KMeans Clustering
      ↓
Model Saving (.pkl)
      ↓
Prediction System (predict.py)
```

---

# Model Evaluation Metrics

* Accuracy
* F1 Score (Macro)
* F1 Score (Weighted)

These metrics ensure balanced evaluation, especially in the presence of class imbalance.

---

# Why These Techniques Were Used

* StandardScaler: Ensures all features are on the same scale for distance-based models
* PCA: Reduces dimensionality and improves clustering performance
* KMeans: Identifies hidden patterns in agricultural conditions
* Random Forest: Handles non-linearity and provides high accuracy
* Label Encoding: Efficient for tree-based models and avoids high dimensionality

---

# Project Structure

```
project/
│
├── data/
│   └── dirty_dataset.csv
│
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   ├── random_forest.pkl
│   ├── logistic.pkl
│   ├── xgboost.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   └── ensemble.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│
└── predict.py
```

---

# How to Run

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the prediction script:

```
python predict.py
```

---

# Prediction Modes

## Mode 1: Crop Prediction

* Input environmental features
* System predicts the most suitable crop

## Mode 2: Crop Insights

* Input crop name
* System provides ideal growing conditions and statistics

---

# Key Highlights

* End-to-end machine learning pipeline
* Combination of supervised and unsupervised learning
* Real-world agricultural application
* Model comparison and optimization
* Deployment-ready prediction system

---

# Future Improvements

* Web-based interface (Streamlit or Flask)
* Integration with real-time weather APIs
* Advanced models like XGBoost or Neural Networks
* Geographic visualization of crop patterns

---

# Summary

This project combines machine learning and clustering techniques to build a practical system for crop recommendation and agricultural analysis using real-world data.
