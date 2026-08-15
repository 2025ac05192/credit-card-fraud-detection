# Credit Card Fraud Detection using Machine Learning

## a. Problem Statement

Credit card fraud is a major challenge in the financial sector because fraudulent transactions can result in significant financial losses for customers and financial institutions. The objective of this project is to develop and compare machine learning classification models for detecting fraudulent credit card transactions.

The project uses the Credit Card Fraud Detection dataset and implements multiple classification algorithms on the same dataset. The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to demonstrate the trained models, compare their performance, visualize evaluation results, and predict whether a transaction is normal or fraudulent.

---

## b. Dataset Description

### Dataset Name

**Credit Card Fraud Detection**

### Dataset Source

Kaggle — Credit Card Fraud Detection dataset by Machine Learning Group - ULB.

Dataset URL:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Characteristics

| Property                |               Value |
| ----------------------- | ------------------: |
| Total Instances         |             284,807 |
| Input Features          |                  30 |
| Target Variable         |               Class |
| Number of Classes       |                   2 |
| Normal Transactions     |             284,315 |
| Fraudulent Transactions |                 492 |
| Fraud Percentage        | Approximately 0.17% |

The dataset contains transactions made by European cardholders during September 2013.

The target variable is `Class`:

* `0` → Normal transaction
* `1` → Fraudulent transaction

The dataset contains the following main features:

* `Time` — elapsed time between transactions
* `V1` to `V28` — anonymized PCA-transformed transaction features
* `Amount` — transaction amount
* `Class` — target variable

The dataset is highly imbalanced because fraudulent transactions represent only approximately 0.17% of all transactions. Therefore, Accuracy alone is not sufficient to determine the best model. Precision, Recall, F1 Score, AUC and MCC are also considered.

### Data Preprocessing

The following preprocessing steps were performed:

1. The target variable `Class` was separated from the input features.
2. The dataset was divided into training and testing sets using an 80:20 split.
3. Stratified sampling was used to preserve the fraud-to-normal transaction ratio in both datasets.
4. The `Time` and `Amount` features were standardized for Logistic Regression and KNN.
5. Tree-based models and Gaussian Naive Bayes were trained using the original feature values.
6. The test set contains 56,962 transactions, including 98 fraudulent transactions.

---

## c. Github Repository Link

**GitHub Repository:**
[Add your GitHub Repository Link here]

The repository contains:

* Complete source code
* Jupyter Notebook containing the experiments and analysis
* Streamlit application
* `requirements.txt`
* `README.md`
* Test data used in the experiments

---

## d. Models Used

The following five classification models were implemented using the same Credit Card Fraud Detection dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)

> **Note:** The assignment wording mentions six models, but the provided model list contains five models. Therefore, this implementation follows the five models explicitly specified in the assignment.

---

## Model Comparison

The following table presents the evaluation results obtained on the test dataset.

| ML Model Name            |     Accuracy |      AUC |    Precision |       Recall |           F1 |          MCC |
| ------------------------ | -----------: | -------: | -----------: | -----------: | -----------: | -----------: |
| Logistic Regression      |     0.999157 | 0.955898 |     0.828947 |     0.642857 |     0.724138 |     0.729596 |
| Decision Tree            |     0.999140 | 0.872238 |     0.752577 |     0.744898 |     0.748718 |     0.748297 |
| KNN                      |     0.999544 | 0.943756 |     0.918605 |     0.806122 |     0.858696 |     0.860305 |
| Gaussian Naive Bayes     |     0.992276 | 0.967731 |     0.137712 |     0.663265 |     0.228070 |     0.299951 |
| Random Forest (Ensemble) | **0.999596** | 0.963027 | **0.941176** | **0.816327** | **0.874317** | **0.876337** |

### Observations About Model Performance

#### Logistic Regression

Logistic Regression achieved an accuracy of 99.9157% and an AUC of 95.59%. Its precision was 82.89%, while recall was 64.29%.

The model performed well overall but missed a relatively larger proportion of fraudulent transactions compared with KNN and Random Forest. Its F1 Score of 72.41% and MCC of 72.96% indicate moderate performance for this highly imbalanced fraud detection problem.

#### Decision Tree

The Decision Tree achieved an accuracy of 99.9140%, with an AUC of 87.22%. It achieved a precision of 75.26% and recall of 74.49%.

Compared with Logistic Regression, the Decision Tree provided slightly better recall but lower AUC and precision. Its F1 Score was 74.87% and MCC was 74.83%, showing reasonable but not the best overall performance.

#### KNN

K-Nearest Neighbors achieved an accuracy of 99.9544%, an AUC of 94.38%, precision of 91.86%, and recall of 80.61%.

Its F1 Score of 85.87% and MCC of 86.03% show that KNN performed very well in identifying fraudulent transactions while maintaining a relatively low number of false positives.

KNN was the second strongest model overall based on the F1 Score and MCC.

#### Gaussian Naive Bayes

Gaussian Naive Bayes achieved the highest AUC of **96.77%**, which was the highest among all five models.

However, its precision was only 13.77%, resulting in a very low F1 Score of 22.81% and MCC of 29.99%. This indicates that the model classified a large number of legitimate transactions as fraudulent.

Therefore, despite having the highest AUC, Gaussian Naive Bayes was not the best model for the final classification decision.

#### Random Forest (Ensemble)

Random Forest achieved the highest overall performance among the five models.

It obtained:

* Accuracy: **99.9596%**
* Precision: **94.12%**
* Recall: **81.63%**
* F1 Score: **87.43%**
* MCC: **87.63%**
* AUC: **96.30%**

Random Forest provided the best balance between detecting fraudulent transactions and avoiding false-positive predictions.

It achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC among the five models. Therefore, Random Forest was selected as the **overall best-performing model** for this dataset.

---

## Overall Winner for the Dataset

### **Random Forest (Ensemble)**

Random Forest is the overall winner because it provides the best balance across the most important classification metrics.

Although Gaussian Naive Bayes achieved the highest AUC, its very low precision and F1 Score indicate that it generated many false-positive predictions.

Random Forest achieved the highest MCC of **0.876337**, which is particularly useful for this highly imbalanced dataset. It also achieved the highest F1 Score of **0.874317**, precision of **0.941176**, and recall of **0.816327**.

Therefore, based on the overall evaluation, **Random Forest is the most suitable model among the evaluated models for this credit card fraud detection problem.**

---

## Streamlit Web Application

An interactive Streamlit application was developed to demonstrate the classification models.

The application provides the following sections:

### 1. Dataset Overview

Displays:

* Number of transactions
* Number of features
* Number of fraudulent transactions
* Fraud percentage
* Dataset preview
* Class distribution
* Dataset statistics

### 2. Model Comparison

Users can compare the five classification models using:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* MCC

### 3. Individual Model Evaluation

Users can select an individual model and view:

* Evaluation metrics
* Confusion matrix
* ROC curve
* AUC score

### 4. Fraud Prediction

Users can select a model and enter transaction feature values. The application predicts whether the transaction is:

* Normal
* Fraudulent

The application also displays the prediction probability.

---

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── notebooks/
    └── Credit_Card_Fraud_Analysis.ipynb
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* KaggleHub
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Credit-Card-Fraud-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in the browser at the local Streamlit address.

---

## Evaluation Metrics

The following metrics were used to evaluate all classification models:

### Accuracy

Measures the overall percentage of correctly classified transactions.

### AUC

Measures the model's ability to distinguish between normal and fraudulent transactions across different classification thresholds.

### Precision

Measures the proportion of transactions predicted as fraudulent that were actually fraudulent.

### Recall

Measures the proportion of actual fraudulent transactions that were correctly detected.

### F1 Score

Provides a balance between Precision and Recall.

### MCC

Matthews Correlation Coefficient provides a balanced measure of classification quality and is particularly useful for highly imbalanced datasets such as credit card fraud detection.

---

## Conclusion

Five machine learning classification models were implemented and evaluated on the Credit Card Fraud Detection dataset.

All models achieved very high accuracy because the dataset is highly imbalanced. Therefore, additional metrics such as Precision, Recall, F1 Score, AUC and MCC were used for a meaningful comparison.

Among the evaluated models, **Random Forest achieved the best overall performance**, with the highest Accuracy, Precision, Recall, F1 Score and MCC.

The project also includes an interactive Streamlit application that allows users to explore the dataset, compare models, view individual model performance, and perform fraud predictions.

