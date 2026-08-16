# Credit Card Fraud Detection using Machine Learning

## a. Problem Statement

Credit card fraud is a major challenge in the financial sector. Detecting fraudulent transactions accurately is important to reduce financial losses and protect customers.

The objective of this project is to implement and compare multiple machine learning classification models for detecting fraudulent credit card transactions.

The models are trained and evaluated using the same Credit Card Fraud Detection dataset.

The following evaluation metrics are used:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive Streamlit web application has also been developed to demonstrate the models, compare their performance, visualize evaluation results, and predict whether a transaction is normal or fraudulent.

---

## b. Dataset Description

### Dataset Name

**Credit Card Fraud Detection**

### Dataset Source

The dataset is obtained from Kaggle.

Kaggle Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Size

| Property | Value |
|---|---:|
| Total Instances | 284,807 |
| Input Features | 30 |
| Target Variable | Class |
| Number of Classes | 2 |
| Normal Transactions | 284,315 |
| Fraudulent Transactions | 492 |

### Target Variable

The target variable is `Class`.

- `0` = Normal transaction
- `1` = Fraudulent transaction

### Features

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`

`V1` to `V28` are anonymized PCA-transformed features.

`Time` represents the time elapsed between transactions.

`Amount` represents the transaction amount.

### Class Imbalance

The dataset is highly imbalanced.

Only approximately 0.17% of the transactions are fraudulent.

Therefore, Accuracy alone is not sufficient to evaluate the models. Precision, Recall, F1 Score, AUC and MCC are also considered.

### Data Preprocessing

The following preprocessing steps were performed:

1. The `Class` column was separated as the target variable.
2. The dataset was divided into training and testing datasets.
3. An 80:20 stratified train-test split was used.
4. Stratification was used to preserve the class distribution in both datasets.
5. `Time` and `Amount` were standardized for Logistic Regression and KNN.
6. The remaining models were trained using the original feature values.

The resulting test dataset contains:

- 56,962 transactions
- 56,864 normal transactions
- 98 fraudulent transactions

---

## c. Github Repository Link

**GitHub Repository:**

https://github.com/2025ac05192/credit-card-fraud-detection

The repository contains:

- Complete Streamlit source code
- Source code for all implemented models
- Saved trained model files
- Test data
- Requirements file
- README documentation

---

## d. Models Used

The following classification models were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)

### Note

The assignment description mentions six models in one section, while the explicitly provided list contains five models. This project implements all five models listed in the required model list.

---

# Model Comparison

All five classification models were evaluated on the same test dataset using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The model comparison below was generated after evaluating the saved trained models on the test dataset.
--------------------------------------------------------------------------------------------------------------
| ML Model Name            | Accuracy     | AUC        | Precision    | Recall     | F1         | MCC        |
|--------------------------|----------    |------------|--------------|------------|------------|------------|
| Logistic Regression      | 0.999157     | 0.955898   | 0.828947     | 0.642857   | 0.724138   | 0.729596   |
| Decision Tree            | 0.999140     | 0.872238   | 0.752577     | 0.744898   | 0.748718   | 0.748297   |
| K-Nearest Neighbor       | 0.999544     | 0.943756   | 0.918605     | 0.806122   | 0.858696   | 0.860305   |
| Gaussian Naive Bayes     | 0.992276     |**0.967731**| 0.137712     | 0.663265   | 0.228070   | 0.299951   |
| Random Forest (Ensemble) | **0.999596** | 0.963027   | **0.941176** |**0.816327**|**0.874317**|**0.876337**|
--------------------------------------------------------------------------------------------------------------
---

# Observations About Model Performance

| ML Model Name | Observation about model performance |                                                                                     
|-------------------------|---|
| **Logistic Regression** |Achieved very high Accuracy of 99.9157% and an AUC of 95.59%. Precision was 82.89%, while Recall was 64.29 %  |was      .                     The model performed well overall but had a comparatively lower Recall and F1 Score than KNN and Random     

Forest.     
The model performed well overall but had a comparatively lower Recall and F1 Score than KNN and Random Forest. |
| **Decision Tree** | Achieved 99.9140% Accuracy and 87.22% AUC. Recall was 74.49% and F1 Score was 74.87%. The model provided reasonable fraud detection performance but had lower AUC and Precision compared with the stronger models. |
| **K-Nearest Neighbor** | Achieved 99.9544% Accuracy, 91.86% Precision, 80.61% Recall, 85.87% F1 Score and 86.03% MCC. It performed strongly and provided a good balance between Precision and Recall. |
| **Gaussian Naive Bayes** | Achieved the highest AUC of 96.77%. However, Precision was only 13.77%, resulting in a low F1 Score of 22.81% and MCC of 29.9951%. This indicates a high number of false-positive predictions despite its strong AUC. |
| **Random Forest (Ensemble)** | Achieved the highest Accuracy (99.9596%), Precision (94.12%), Recall (81.63%), F1 Score (87.43%) and MCC (87.63%). It provided the best overall balance between detecting fraudulent transactions and minimizing false-positive predictions. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** was the overall best-performing model. It achieved the highest Accuracy, Precision, Recall, F1 Score and MCC among all five models. |


--------------------------------------------------------------------------------------------------------------------------------------------
|  ML Model Name                 |                         Observation about Model Performance  | |
---------------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression | Achieved very high Accuracy of 99.9157% and an AUC of 95.59%. Precision was 82.89%, while Recall was 64.29%. The model performed well overall but had comparatively lower Recall and F1 Score than KNN and Random Forest.
--------------------------------------------------------------------------------------------------------------------------------------------
Decision Tree       | Achieved 99.9140% Accuracy and 87.22% AUC. Recall was 74.49% and F1 Score was 74.87%. It provided reasonable fraud detection performance but had lower AUC and Precision compared with the stronger models.
--------------------------------------------------------------------------------------------------------------------------------------------
K-Nearest Neighbor   | Achieved 99.9544% Accuracy, 91.86% Precision, 80.61% Recall, 85.87% F1 Score and 86.03% MCC. It performed strongly and provided a good balance between Precision and Recall.
--------------------------------------------------------------------------------------------------------------------------------------------
Gaussian Naive Bayes | Achieved the highest AUC of 96.77%. However, Precision was only 13.77%, resulting in a low F1 Score of 22.81% and MCC of 29.9951%. This indicates a high number of false-positive predictions despite its strong AUC.
---------------------------------------------------------------------------------------------------------------------------------------------
Random Forest (Ensemble) |Achieved the highest Accuracy (99.9596%), Precision (94.12%), Recall (81.63%), F1 Score (87.43%) and MCC (87.63%). It provided the best overall balance between detecting fraudulent transactions and minimizing false-positive predictions.
-----------------------------------------------------------------------------------------------------------------------------------------------
---


# Observations About Model Performance

| ML Model Name           | Observation about model performance                                                                            |
|-------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Achieved very high Accuracy of 99.9157% and an AUC of 95.59%. Precision was 82.89%, while Recall was 64.29%. The model performed well overall but had a comparatively lower Recall and F1 Score than KNN and Random Forest.                                     |
-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Decision Tree** | Achieved 99.9140% Accuracy and 87.22% AUC. Recall was 74.49% and F1 Score was 74.87%. The model provided reasonable fraud detection performance but had lower AUC and Precision compared with the stronger models.                                                       |
-----------------------------------------------------------------------------------------------------------------------------------------------|
| **K-Nearest Neighbor** | Achieved 99.9544% Accuracy, 91.86% Precision, 80.61% Recall, 85.87% F1 Score and 86.03% MCC. It performed strongly and provided a good balance between Precision and Recall.                                                                                      |
-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Gaussian Naive Bayes** | Achieved the highest AUC of 96.77%. However, Precision was only 13.77%, resulting in a low F1 Score of 22.81% and MCC of 29.9951%. This indicates a high number of false-positive predictions despite its strong AUC.                                            |
-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Random Forest (Ensemble)** | Achieved the highest Accuracy (99.9596%), Precision (94.12%), Recall (81.63%), F1 Score (87.43%) and MCC (87.63%). It provided the best overall balance between detecting fraudulent transactions and minimizing false-positive predictions.                |
-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** was the overall best-performing model. It achieved the highest Accuracy, Precision, Recall, F1 Score and MCC among all five models.                                                                                     |
-----------------------------------------------------------------------------------------------------------------------------------------------|

# Overall Winner

## Random Forest (Ensemble)

Based on the evaluation results, **Random Forest** was selected as the overall best-performing model for the Credit Card Fraud Detection dataset.

Its performance was:

| Metric | Score |
|---|---:|
| Accuracy | **0.999596** |
| AUC | 0.963027 |
| Precision | **0.941176** |
| Recall | **0.816327** |
| F1 Score | **0.874317** |
| MCC | **0.876337** |

Random Forest achieved the highest Accuracy, Precision, Recall, F1 Score and MCC among the five implemented models.

Although Gaussian Naive Bayes achieved the highest AUC of 0.967731, its Precision, F1 Score and MCC were considerably lower. Therefore, AUC alone was not used to determine the overall winner.

Since the dataset is highly imbalanced, **F1 Score, Precision, Recall and MCC** are particularly important when selecting the most suitable fraud detection model.

Therefore, **Random Forest (Ensemble) is considered the overall winner for this dataset.**
---

# Streamlit Web Application

An interactive Streamlit application was developed to demonstrate the classification models.

The application contains the following sections.

## 1. Dataset Overview

The application displays:

- Total number of transactions
- Number of input features
- Number of fraudulent transactions
- Fraud percentage
- Dataset preview
- Class distribution
- Train-test split information
- Dataset statistics
- Download option for `test_data.csv`

## 2. Model Comparison

Users can compare all five models using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

The application also provides a visual comparison of the selected metric.

## 3. Individual Model Evaluation

Users can select an individual model and view:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- True Positive
- True Negative
- False Positive
- False Negative
- ROC Curve

## 4. Fraud Prediction

Users can select a classification model and perform a transaction prediction.

The application supports:

- Selecting a sample transaction from the test dataset
- Entering transaction feature values manually
- Predicting Normal or Fraudulent transaction
- Displaying prediction probabilities
- Comparing actual and predicted classes for test transactions

---

# Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.py
    ├── logistic_regression.pkl
    │
    ├── decision_tree.py
    ├── decision_tree.pkl
    │
    ├── knn.py
    ├── knn.pkl
    │
    ├── naive_bayes.py
    ├── naive_bayes.pkl
    │
    ├── random_forest.py
    ├── random_forest.pkl
    │
    └── scaler.pkl