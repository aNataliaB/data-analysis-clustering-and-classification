# Student Dropout and Academic Success Prediction
A machine learning project focused on predicting student dropout and academic success using classification techniques. The goal is to identify students at risk early in their academic career to enable timely interventions.

## Project Overview
The project involves automated data preprocessing, exploratory data analysis (EDA), and training multiple classification models to accurately predict student outcomes based on demographic, socioeconomic, and academic data.

### Key Features:
* **Exploratory Data Analysis:** Generating descriptive statistics, box-plots for outlier detection, and empirical histograms with density curves.
* **Data Preprocessing:** Logarithmic transformations to handle skewness (e.g., for age distribution) and data standardization using `StandardScaler`.
* **Predictive Modeling:** Implementing and evaluating multiple algorithms:
  * Logistic Regression (with balanced class weights)
  * Random Forest Classifier
  * Support Vector Machine (SVM)
  * Hybrid Ensemble Model (Weighted average of model probabilities)
* **Model Evaluation:** Using 5-Fold Cross-Validation, Accuracy, F1-Score, Classification Reports, and Confusion Matrices to compare performance.
* **Feature Importance:** Extracting and visualizing key features driving student dropout using Random Forest.

## Team & Collaboration
This project was developed as a collaborative academic assignment.
* **My Contribution:** Responsible for implementing the complete Python end-to-end pipeline (`main.py`), performing data preprocessing, building the hybrid ensemble model, creating data visualizations (Seaborn/Matplotlib), and co-authoring the final analytical report.

## Technologies & Libraries
* **Language:** Python 
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## Repository Structure
* `main.py` - Core Python script containing data processing and machine learning workflows.
* `data.csv` - The dataset containing student records.
* `Przewidywanie_rezygnacji_ze_studiów.pdf` - Full written analytical report (in Polish).
