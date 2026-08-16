import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    ### Machine Learning Classification Dashboard

    This interactive application demonstrates five machine
    learning classification models for detecting fraudulent
    credit card transactions.

    **Implemented Models:**

    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbor Classifier
    - Gaussian Naive Bayes Classifier
    - Random Forest Classifier (Ensemble Model)
    """
)


# ============================================================
# SIDEBAR - TEST DATA UPLOAD
# ============================================================

st.sidebar.header("📂 Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload only the test dataset CSV."
)


if uploaded_file is None:

    st.info(
        """
        ### 📂 Upload Test Data

        Please upload the `test_data.csv` file using the
        file uploader in the sidebar.

        **Note:** Only the test dataset needs to be uploaded.
        The complete Kaggle dataset is not required by this
        Streamlit application.
        """
    )

    st.stop()


# ============================================================
# LOAD UPLOADED TEST DATA
# ============================================================

try:

    uploaded_test_data = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE TEST DATA
# ============================================================

if "Class" not in uploaded_test_data.columns:

    st.error(
        """
        ❌ Invalid CSV file.

        The uploaded test dataset must contain a
        `Class` column.
        """
    )

    st.stop()


X_test = uploaded_test_data.drop(
    "Class",
    axis=1
)

y_test = uploaded_test_data["Class"]


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

if X_test.shape[1] != 30:

    st.error(
        f"""
        ❌ Invalid number of features.

        Expected: 30 features

        Found: {X_test.shape[1]} features
        """
    )

    st.stop()


# ============================================================
# VALIDATE CLASS VALUES
# ============================================================

unique_classes = set(
    y_test.unique()
)


if not unique_classes.issubset({0, 1}):

    st.error(
        """
        ❌ Invalid target values.

        The `Class` column must contain only:
        0 = Normal
        1 = Fraud
        """
    )

    st.stop()


st.sidebar.success(
    f"✅ Test data loaded: {len(uploaded_test_data):,} rows"
)


# ============================================================
# LOAD SAVED SCALER
# ============================================================

scaler_path = os.path.join(
    "model",
    "scaler.pkl"
)


if not os.path.exists(
    scaler_path
):

    st.error(
        """
        ❌ `scaler.pkl` was not found inside the
        `model` folder.

        Please make sure the saved scaler is included
        in the GitHub repository.
        """
    )

    st.stop()


scaler = joblib.load(
    scaler_path
)


# ============================================================
# SCALE TEST DATA
# ============================================================

X_test_scaled = X_test.copy()


X_test_scaled[
    ["Time", "Amount"]
] = scaler.transform(
    X_test[
        ["Time", "Amount"]
    ]
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    model_files = {

        "Logistic Regression":
            "model/logistic_regression.pkl",

        "Decision Tree":
            "model/decision_tree.pkl",

        "K-Nearest Neighbor":
            "model/knn.pkl",

        "Gaussian Naive Bayes":
            "model/naive_bayes.pkl",

        "Random Forest (Ensemble)":
            "model/random_forest.pkl"
    }


    loaded_models = {}


    for model_name, model_path in model_files.items():

        if not os.path.exists(
            model_path
        ):

            st.error(
                f"Model file not found: {model_path}"
            )

            st.stop()


        loaded_models[
            model_name
        ] = joblib.load(
            model_path
        )


    return loaded_models


models = load_models()


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model
):

    # --------------------------------------------------------
    # MODELS USING SCALED FEATURES
    # --------------------------------------------------------

    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbor"
    ]:

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]


    # --------------------------------------------------------
    # MODELS USING ORIGINAL FEATURES
    # --------------------------------------------------------

    else:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }


    return (
        metrics,
        predictions,
        probabilities
    )


# ============================================================
# DYNAMIC OBSERVATION GENERATOR
# ============================================================

def generate_model_observation(
    model_name,
    row,
    results_df
):

    observations = []


    # --------------------------------------------------------
    # FIND BEST METRICS
    # --------------------------------------------------------

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]


    best_metrics = []


    for metric in metric_columns:

        if row[metric] == results_df[
            metric
        ].max():

            best_metrics.append(
                metric
            )


    if best_metrics:

        observations.append(
            "Best in "
            + ", ".join(best_metrics)
        )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision_median = results_df[
        "Precision"
    ].median()


    if row["Precision"] >= precision_median:

        observations.append(
            "strong Precision"
        )

    else:

        observations.append(
            "lower Precision"
        )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall_median = results_df[
        "Recall"
    ].median()


    if row["Recall"] >= recall_median:

        observations.append(
            "strong Recall"
        )

    else:

        observations.append(
            "lower Recall"
        )


    # --------------------------------------------------------
    # F1 SCORE
    # --------------------------------------------------------

    f1_median = results_df[
        "F1 Score"
    ].median()


    if row["F1 Score"] >= f1_median:

        observations.append(
            "good balance between Precision and Recall"
        )

    else:

        observations.append(
            "weaker balance between Precision and Recall"
        )


    # --------------------------------------------------------
    # FALSE POSITIVE RISK
    # --------------------------------------------------------

    if row["Precision"] < 0.50:

        observations.append(
            "higher risk of false-positive predictions"
        )


    # --------------------------------------------------------
    # FRAUD DETECTION
    # --------------------------------------------------------

    if row["Recall"] < 0.70:

        observations.append(
            "lower fraud detection capability"
        )


    return (
        ". ".join(observations)
        + "."
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "📌 Navigation"
)


page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Model Comparison",
        "Individual Model",
        "Fraud Prediction"
    ]
)


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header(
        "📊 Dataset Overview"
    )


    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Test Transactions",
        f"{len(uploaded_test_data):,}"
    )


    col2.metric(
        "Input Features",
        X_test.shape[1]
    )


    col3.metric(
        "Fraud Transactions",
        f"{int(y_test.sum()):,}"
    )


    col4.metric(
        "Fraud Percentage",
        f"{y_test.mean() * 100:.3f}%"
    )


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Test Dataset"
    )


    st.write(
        """
        The uploaded CSV represents the test dataset used
        to evaluate all five trained classification models.

        The dataset contains 30 input features and one target
        variable called `Class`.
        """
    )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Test Dataset Preview"
    )


    st.dataframe(
        uploaded_test_data.head(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Class Distribution"
    )


    class_counts = y_test.value_counts()


    class_distribution = pd.DataFrame({

        "Transaction Type": [
            "Normal",
            "Fraud"
        ],

        "Count": [
            int(
                class_counts.get(
                    0,
                    0
                )
            ),

            int(
                class_counts.get(
                    1,
                    0
                )
            )
        ]
    })


    st.bar_chart(
        class_distribution.set_index(
            "Transaction Type"
        )
    )


    st.dataframe(
        class_distribution,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Dataset Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Number of Rows:**",
            len(uploaded_test_data)
        )

        st.write(
            "**Number of Features:**",
            X_test.shape[1]
        )


    with col2:

        st.write(
            "**Normal Transactions:**",
            int(
                class_counts.get(
                    0,
                    0
                )
            )
        )

        st.write(
            "**Fraudulent Transactions:**",
            int(
                class_counts.get(
                    1,
                    0
                )
            )
        )


    # --------------------------------------------------------
    # DOWNLOAD / REUSE UPLOADED DATA
    # --------------------------------------------------------

    st.subheader(
        "Download Uploaded Test Data"
    )


    st.download_button(
        label="📥 Download Test Data",
        data=uploaded_test_data.to_csv(
            index=False
        ),
        file_name="test_data.csv",
        mime="text/csv"
    )


# ============================================================
# 2. MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header(
        "📈 Model Performance Comparison"
    )


    st.write(
        """
        All five classification models are evaluated on
        the uploaded test dataset using the six required
        evaluation metrics.
        """
    )


    # --------------------------------------------------------
    # CALCULATE RESULTS
    # --------------------------------------------------------

    results = []


    for model_name, model in models.items():

        metrics, _, _ = evaluate_model(
            model_name,
            model
        )


        results.append({

            "Model": model_name,

            **metrics
        })


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # EVALUATION METRICS TABLE
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )


    display_results = results_df.copy()


    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]


    for column in metric_columns:

        display_results[column] = (
            display_results[column]
            .map(
                lambda value:
                f"{value:.6f}"
            )
        )


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # METRIC SELECTION
    # --------------------------------------------------------

    st.subheader(
        "Metric-wise Model Comparison"
    )


    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        metric_columns
    )


    chart_data = results_df[
        [
            "Model",
            selected_metric
        ]
    ].set_index(
        "Model"
    )


    st.bar_chart(
        chart_data
    )


    # --------------------------------------------------------
    # BEST MODEL FOR SELECTED METRIC
    # --------------------------------------------------------

    best_index = results_df[
        selected_metric
    ].idxmax()


    best_model = results_df.loc[
        best_index,
        "Model"
    ]


    best_score = results_df.loc[
        best_index,
        selected_metric
    ]


    st.success(
        f"Best model based on {selected_metric}: "
        f"{best_model} "
        f"({best_score:.6f})"
    )


    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    st.subheader(
        "Observations About Model Performance"
    )


    observations = []


    for _, row in results_df.iterrows():

        observation = generate_model_observation(
            row["Model"],
            row,
            results_df
        )


        observations.append({

            "ML Model Name":
                row["Model"],

            "Observation":
                observation
        })


    observations_df = pd.DataFrame(
        observations
    )


    st.dataframe(
        observations_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # OVERALL WINNER
    # --------------------------------------------------------

    st.subheader(
        "Overall Winner for Your Dataset"
    )


    results_df["Overall Score"] = (
        results_df["F1 Score"]
        +
        results_df["MCC"]
    ) / 2


    winner_index = results_df[
        "Overall Score"
    ].idxmax()


    overall_winner = results_df.loc[
        winner_index,
        "Model"
    ]


    winner_f1 = results_df.loc[
        winner_index,
        "F1 Score"
    ]


    winner_mcc = results_df.loc[
        winner_index,
        "MCC"
    ]


    winner_score = results_df.loc[
        winner_index,
        "Overall Score"
    ]


    st.success(
        f"🏆 Overall Winner: {overall_winner}"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "F1 Score",
        f"{winner_f1:.6f}"
    )


    col2.metric(
        "MCC",
        f"{winner_mcc:.6f}"
    )


    col3.metric(
        "Combined Score",
        f"{winner_score:.6f}"
    )


    st.info(
        """
        The overall comparison considers F1 Score and MCC
        because the fraud dataset is highly imbalanced.
        Accuracy alone is not sufficient for selecting the
        best fraud detection model.
        """
    )


# ============================================================
# 3. INDIVIDUAL MODEL
# ============================================================

elif page == "Individual Model":

    st.header(
        "🔍 Individual Model Evaluation"
    )


    # --------------------------------------------------------
    # MODEL DROPDOWN
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )


    model = models[
        selected_model
    ]


    metrics, predictions, probabilities = evaluate_model(
        selected_model,
        model
    )


    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    st.subheader(
        f"{selected_model} - Evaluation Metrics"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.6f}"
    )


    col2.metric(
        "AUC",
        f"{metrics['AUC']:.6f}"
    )


    col3.metric(
        "Precision",
        f"{metrics['Precision']:.6f}"
    )


    col4, col5, col6 = st.columns(3)


    col4.metric(
        "Recall",
        f"{metrics['Recall']:.6f}"
    )


    col5.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.6f}"
    )


    col6.metric(
        "MCC",
        f"{metrics['MCC']:.6f}"
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    tn, fp, fn, tp = cm.ravel()


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    image = ax.imshow(
        cm
    )


    ax.set_title(
        f"{selected_model} - Confusion Matrix"
    )


    ax.set_xlabel(
        "Predicted Class"
    )


    ax.set_ylabel(
        "Actual Class"
    )


    ax.set_xticks(
        [0, 1]
    )


    ax.set_yticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "Normal",
            "Fraud"
        ]
    )


    ax.set_yticklabels(
        [
            "Normal",
            "Fraud"
        ]
    )


    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    fig.colorbar(
        image,
        ax=ax
    )


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX VALUES
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix Details"
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "True Negative",
        int(tn)
    )


    col2.metric(
        "False Positive",
        int(fp)
    )


    col3.metric(
        "False Negative",
        int(fn)
    )


    col4.metric(
        "True Positive",
        int(tp)
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "Classification Report"
    )


    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Fraud"
        ],
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df.style.format(
            "{:.4f}"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader(
        "ROC Curve"
    )


    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {metrics['AUC']:.6f}"
    )


    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )


    ax.set_xlabel(
        "False Positive Rate"
    )


    ax.set_ylabel(
        "True Positive Rate"
    )


    ax.set_title(
        f"{selected_model} - ROC Curve"
    )


    ax.legend()


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


# ============================================================
# 4. FRAUD PREDICTION
# ============================================================

elif page == "Fraud Prediction":

    st.header(
        "💳 Fraud Transaction Prediction"
    )


    st.info(
        """
        Select a trained model and test a transaction from
        the uploaded test dataset or enter feature values
        manually.
        """
    )


    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )


    model = models[
        selected_model
    ]


    # --------------------------------------------------------
    # INPUT METHOD
    # --------------------------------------------------------

    input_method = st.radio(
        "Choose Input Method",
        [
            "Use Sample Test Transaction",
            "Enter Values Manually"
        ]
    )


    # ========================================================
    # SAMPLE TEST TRANSACTION
    # ========================================================

    if input_method == "Use Sample Test Transaction":

        transaction_type = st.radio(
            "Select Transaction Type",
            [
                "Normal",
                "Fraud"
            ]
        )


        if transaction_type == "Normal":

            available_indices = y_test[
                y_test == 0
            ].index

        else:

            available_indices = y_test[
                y_test == 1
            ].index


        if len(available_indices) == 0:

            st.warning(
                "No transactions of the selected class "
                "are available in the uploaded test data."
            )

            st.stop()


        max_samples = min(
            20,
            len(available_indices)
        )


        sample_number = st.number_input(
            "Sample Number",
            min_value=1,
            max_value=max_samples,
            value=1,
            step=1
        )


        selected_index = available_indices[
            int(sample_number) - 1
        ]


        input_df = X_test.loc[
            [selected_index]
        ].copy()


        actual_class = int(
            y_test.loc[
                selected_index
            ]
        )


        st.subheader(
            "Selected Transaction"
        )


        st.dataframe(
            input_df,
            use_container_width=True
        )


        st.write(
            "Actual Class:",
            "Fraud"
            if actual_class == 1
            else "Normal"
        )


    # ========================================================
    # MANUAL INPUT
    # ========================================================

    else:

        actual_class = None


        st.subheader(
            "Enter Transaction Features"
        )


        input_values = {}


        col1, col2 = st.columns(2)


        for index, feature in enumerate(
            X_test.columns
        ):

            default_value = float(
                X_test[feature].median()
            )


            if index % 2 == 0:

                input_values[
                    feature
                ] = col1.number_input(
                    feature,
                    value=default_value,
                    format="%.6f"
                )

            else:

                input_values[
                    feature
                ] = col2.number_input(
                    feature,
                    value=default_value,
                    format="%.6f"
                )


        input_df = pd.DataFrame(
            [input_values]
        )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Transaction",
        type="primary"
    ):


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        input_scaled = input_df.copy()


        input_scaled[
            ["Time", "Amount"]
        ] = scaler.transform(
            input_df[
                ["Time", "Amount"]
            ]
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        if selected_model in [
            "Logistic Regression",
            "K-Nearest Neighbor"
        ]:

            prediction = model.predict(
                input_scaled
            )[0]


            probability = model.predict_proba(
                input_scaled
            )[0]


        else:

            prediction = model.predict(
                input_df
            )[0]


            probability = model.predict_proba(
                input_df
            )[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION"
            )

        else:

            st.success(
                "✅ NORMAL TRANSACTION"
            )


        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        if actual_class is not None:

            actual_label = (
                "Fraud"
                if actual_class == 1
                else "Normal"
            )


            predicted_label = (
                "Fraud"
                if prediction == 1
                else "Normal"
            )


            comparison_df = pd.DataFrame({

                "Result": [
                    "Actual",
                    "Predicted"
                ],

                "Class": [
                    actual_label,
                    predicted_label
                ]
            })


            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


            if prediction == actual_class:

                st.success(
                    "The model prediction matches "
                    "the actual class."
                )

            else:

                st.warning(
                    "The model prediction does not "
                    "match the actual class."
                )


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probability"
        )


        probability_df = pd.DataFrame({

            "Class": [
                "Normal",
                "Fraud"
            ],

            "Probability": [
                probability[0],
                probability[1]
            ]
        })


        st.dataframe(
            probability_df.style.format({
                "Probability": "{:.6f}"
            }),
            use_container_width=True,
            hide_index=True
        )


        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )