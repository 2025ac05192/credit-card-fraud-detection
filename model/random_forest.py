import os
import joblib
import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "random_forest.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    "Random Forest model saved successfully."
)