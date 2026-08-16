import os
import joblib
import kagglehub
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
    roc_curve
)

from sklearn.model_selection import train_test_split


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

    **Models implemented:**

    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbor Classifier
    - Gaussian Naive Bayes Classifier
    - Random Forest Classifier
    """
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    path = kagglehub.dataset_download(
        "mlg-ulb/creditcardfraud"
    )

    file_path = os.path.join(
        path,
        "creditcard.csv"
    )

    data = pd.read_csv(
        file_path
    )

    return data


with st.spinner("Loading Credit Card Fraud dataset..."):

    df = load_dataset()


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# CREATE TEST DATA
# ============================================================

test_data = X_test.copy()

test_data["Class"] = y_test.values

test_csv = test_data.to_csv(
    index=False
)


# ============================================================
# LOAD SAVED SCALER
# ============================================================

scaler_path = os.path.join(
    "model",
    "scaler.pkl"
)


if not os.path.exists(scaler_path):

    st.error(
        "scaler.pkl was not found inside the model folder."
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

        "K-Nearest Neighbors":
            "model/knn.pkl",

        "Gaussian Naive Bayes":
            "model/naive_bayes.pkl",

        "Random Forest":
            "model/random_forest.pkl"
    }


    loaded_models = {}


    for model_name, model_path in model_files.items():

        if not os.path.exists(model_path):

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

    # Logistic Regression and KNN
    # were trained using scaled data.

    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbors"
    ]:

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

    # Decision Tree, Naive Bayes and
    # Random Forest use original features.

    else:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


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
# GENERATE DYNAMIC MODEL OBSERVATION
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

    metric_columns = {

        "Accuracy": "Accuracy",

        "AUC": "AUC",

        "Precision": "Precision",

        "Recall": "Recall",

        "F1 Score": "F1 Score",

        "MCC": "MCC"
    }


    best_metrics = []


    for metric_name, column in metric_columns.items():

        if row[column] == results_df[column].max():

            best_metrics.append(
                metric_name
            )


    # --------------------------------------------------------
    # BEST METRIC OBSERVATION
    # --------------------------------------------------------

    if best_metrics:

        observations.append(
            "Best in "
            + ", ".join(best_metrics)
        )


    # --------------------------------------------------------
    # PRECISION OBSERVATION
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
    # RECALL OBSERVATION
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
    # F1 SCORE OBSERVATION
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
    # FALSE POSITIVE OBSERVATION
    # --------------------------------------------------------

    if row["Precision"] < 0.50:

        observations.append(
            "higher risk of false-positive predictions"
        )


    # --------------------------------------------------------
    # FRAUD DETECTION OBSERVATION
    # --------------------------------------------------------

    if row["Recall"] < 0.70:

        observations.append(
            "lower fraud detection capability"
        )


    # --------------------------------------------------------
    # RETURN OBSERVATION
    # --------------------------------------------------------

    return (
        ". ".join(observations)
        + "."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "💳 Navigation"
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


st.sidebar.markdown("---")

st.sidebar.write(
    "Credit Card Fraud Detection"
)

st.sidebar.write(
    "Machine Learning Classification Project"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header(
        "📊 Dataset Overview"
    )


    # --------------------------------------------------------
    # KEY DATASET METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Transactions",
        f"{len(df):,}"
    )


    col2.metric(
        "Input Features",
        X.shape[1]
    )


    col3.metric(
        "Fraud Transactions",
        f"{int(y.sum()):,}"
    )


    col4.metric(
        "Fraud Percentage",
        f"{y.mean() * 100:.3f}%"
    )


    # --------------------------------------------------------
    # DATASET DESCRIPTION
    # --------------------------------------------------------

    st.subheader(
        "Dataset Description"
    )

    st.write(
        """
        The Credit Card Fraud Detection dataset contains
        transactions made by European cardholders.

        The dataset contains 284,807 transactions and
        30 input features.

        The target variable is:

        • 0 = Normal transaction

        • 1 = Fraudulent transaction

        The dataset is highly imbalanced because fraudulent
        transactions represent approximately 0.17% of all
        transactions.
        """
    )


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Class Distribution"
    )


    class_counts = df[
        "Class"
    ].value_counts()


    class_df = pd.DataFrame({

        "Transaction Type": [
            "Normal",
            "Fraud"
        ],

        "Count": [
            int(class_counts.get(0, 0)),
            int(class_counts.get(1, 0))
        ]
    })


    st.bar_chart(
        class_df.set_index(
            "Transaction Type"
        )
    )


    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    st.subheader(
        "Train / Test Split"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "Training Records",
        f"{len(X_train):,}"
    )


    col2.metric(
        "Testing Records",
        f"{len(X_test):,}"
    )


    st.write(
        """
        An 80:20 stratified train-test split was used.
        Stratification preserves the proportion of normal
        and fraudulent transactions in both datasets.
        """
    )


    # --------------------------------------------------------
    # TEST CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Test Data Distribution"
    )


    test_class_counts = y_test.value_counts()


    test_distribution = pd.DataFrame({

        "Transaction Type": [
            "Normal",
            "Fraud"
        ],

        "Count": [
            int(test_class_counts.get(0, 0)),
            int(test_class_counts.get(1, 0))
        ]
    })


    st.dataframe(
        test_distribution,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD TEST DATA
    # --------------------------------------------------------

    st.subheader(
        "Download Test Data"
    )


    st.write(
        """
        Download the test dataset used for evaluating
        all five classification models.
        """
    )


    st.download_button(
        label="📥 Download test_data.csv",
        data=test_csv,
        file_name="test_data.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "Dataset Statistics"
    )


    st.dataframe(
        df.describe().T,
        use_container_width=True
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header(
        "📈 Model Performance Comparison"
    )


    st.write(
        """
        All five classification models are evaluated on
        the same test dataset using the required metrics.
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
    # MODEL COMPARISON TABLE
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )


    formatted_results = results_df.copy()


    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]


    for column in metric_columns:

        formatted_results[column] = (
            formatted_results[column]
            .map(
                lambda x: f"{x:.6f}"
            )
        )


    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # METRIC COMPARISON CHART
    # --------------------------------------------------------

    st.subheader(
        "Compare Models"
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
    # DYNAMIC OBSERVATIONS
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

            "Observation about model performance":
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


    # F1 Score and MCC are used as the
    # primary indicators because the
    # dataset is highly imbalanced.

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


    st.write(
        f"""
        Based on the combined F1 Score and MCC,
        **{overall_winner}** achieved the strongest
        overall classification performance.

        **F1 Score:** {winner_f1:.6f}

        **MCC:** {winner_mcc:.6f}

        **Combined F1/MCC Score:** {winner_score:.6f}
        """
    )


    st.info(
        """
        Since the dataset is highly imbalanced, Accuracy
        alone should not be used to select the best model.
        Precision, Recall, F1 Score, AUC and MCC are also
        considered.
        """
    )


# ============================================================
# INDIVIDUAL MODEL
# ============================================================

elif page == "Individual Model":

    st.header(
        "🔍 Individual Model Evaluation"
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


    metrics, predictions, probabilities = evaluate_model(
        selected_model,
        model
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        f"{selected_model} Performance"
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
# FRAUD PREDICTION
# ============================================================

elif page == "Fraud Prediction":

    st.header(
        "💳 Fraud Transaction Prediction"
    )


    st.info(
        """
        The V1 to V28 attributes are anonymized
        PCA-transformed features from the original dataset.

        You can either select a real transaction from the
        test dataset or enter feature values manually.
        """
    )


    # --------------------------------------------------------
    # SELECT MODEL
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
            X.columns
        ):

            default_value = float(
                X[feature].median()
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
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Transaction",
        type="primary"
    ):


        # ----------------------------------------------------
        # SCALE INPUT FOR LOGISTIC REGRESSION / KNN
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
        # MAKE PREDICTION
        # ----------------------------------------------------

        if selected_model in [
            "Logistic Regression",
            "K-Nearest Neighbors"
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
        # DISPLAY PREDICTION
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