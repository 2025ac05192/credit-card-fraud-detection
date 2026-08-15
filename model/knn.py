import os
import joblib
import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# LOAD DATASET
# ============================================================

path = kagglehub.dataset_download(
    "mlg-ulb/creditcardfraud"
)

file_path = os.path.join(
    path,
    "creditcard.csv"
)

df = pd.read_csv(file_path)


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
# SCALE TIME AND AMOUNT
# ============================================================

scaler = StandardScaler()

X_train_scaled = X_train.copy()

X_train_scaled[
    ["Time", "Amount"]
] = scaler.fit_transform(
    X_train[
        ["Time", "Amount"]
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "knn.pkl"
)

joblib.dump(
    model,
    model_path
)


print(
    "KNN model saved successfully."
)