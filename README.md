# Diabetes-dataset-EDA-ML-TABLEAU-

# 🩺 Diabetes Prediction & Analysis with Machine Learning and Tableau

## Overview

This project explores and predicts diabetes occurrence using a combination of exploratory data analysis (EDA), machine learning models, and interactive visual analytics with Tableau. The workflow demonstrates a complete data science pipeline from raw data to actionable insights and model evaluation.

---

## Table of Contents

- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance Metrics](#model-performance-metrics)
- [Tableau Dashboard](#tableau-dashboard)
- [How to Run](#how-to-run)
- [Results & Insights](#results--insights)

---

## Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:** Age, Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Outcome (diabetes: 0/1)

---

## Exploratory Data Analysis (EDA)

EDA was performed using Python (pandas, matplotlib, seaborn):

- **Missing Values:** Checked and handled (imputed zeros or missing values for key features).
- **Distribution Analysis:** Plotted histograms and boxplots for Glucose, BMI, Age, Insulin, etc.
- **Correlation Analysis:** Visualized relationships between features (correlation heatmap).
- **Outcome Proportion:** Compared feature distributions between diabetic (Outcome=1) and non-diabetic (Outcome=0) groups.
- **Feature Engineering:** Created age bins, normalized features, and explored relationships (e.g., Age vs. Glucose, BMI by Outcome).

---

## Machine Learning Models

Trained and evaluated multiple supervised ML algorithms using scikit-learn:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

**Pipeline:**
- Split data into training and test sets
- Feature scaling/normalization as appropriate
- Model fitting and hyperparameter tuning (grid search where relevant)
- Generated predictions and prediction probabilities

---

## Model Performance Metrics

Evaluated each model using key metrics:

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  |   XX     |   XX      |  XX    |   XX     |   XX    |
| Random Forest        |   XX     |   XX      |  XX    |   XX     |   XX    |
| SVM                  |   XX     |   XX      |  XX    |   XX     |   XX    |

- **Confusion Matrix:** Assessed types of errors (false positives, false negatives)
- **ROC Curve:** Compared models using ROC-AUC
- **Error Analysis:** Examined false positive/false negative counts per model

_(Actual numbers are in the Jupyter notebook and Tableau dashboard)_

---

## Tableau Dashboard

📊 **[🌟 View Interactive Tableau Dashboard](https://public.tableau.com/views/DiabetespredictionDataanalysisandML_17511820771820/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

Features in the dashboard:

- **Model Comparison:** Bar chart & heatmap of model metrics (accuracy, precision, recall, F1, ROC AUC)
- **Error Analysis:** Visualization of false positives and false negatives per model
- **Feature Distributions:** Explore Glucose, BMI, and Age distributions by diabetes status
- **Interactive Filtering:** Filter and explore data and predictions by age group, outcome, or model

---

## How to Run

1. **Clone this repo:**
    ```bash
    git clone https://github.com/yourusername/diabetes-ml-tableau.git
    cd diabetes-ml-tableau
    ```

2. **Open and run the Jupyter notebook (`diabetes_analysis.ipynb`):**
    - Performs EDA
    - Trains and evaluates ML models
    - Exports predictions and metrics for Tableau

3. **Tableau Visualization:**
    - Tableau workbook (`.twbx`) and CSVs are provided
    - Or view the interactive dashboard [here](https://public.tableau.com/views/DiabetespredictionDataanalysisandML_17511820771820/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Results & Insights

- Random Forest achieved the highest overall performance based on ROC AUC and F1 Score.
- Glucose and BMI are the most significant predictors of diabetes in the dataset.
- Error analysis shows [example: SVM had more false negatives, while Logistic Regression had more false positives].
- Tableau dashboard provides an interactive platform for further exploration and communication of findings.

---

## Credits

- Dataset: UCI / Kaggle Pima Indians Diabetes
- Analysis & ML: Python (pandas, scikit-learn, matplotlib, seaborn)
- Visualization: Tableau Public

---

## Contact

Questions? Suggestions?  
Open an issue or contact [your.email@example.com](mailto:your.email@example.com).

