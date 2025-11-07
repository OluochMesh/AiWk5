import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df.replace("?", np.nan, inplace=True)

    df.drop(columns=["weight", "patient_nbr"], inplace=True, errors="ignore")

    df["readmitted"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    return df


def split_features_labels(df):
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    return X, y


def build_preprocessing_pipeline(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


def train_model():
    df = load_data("data/readmission.csv")
    df = clean_data(df)

    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    print("Positive class rate (train):", sum(y_train) / len(y_train))

    preprocessor = build_preprocessing_pipeline(X)

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_encoded, y_train)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        use_label_encoder=False,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("✅ Training model...")
    model.fit(X_train_res, y_train_res)

    print("✅ Evaluating model...")
    y_pred = model.predict(X_test_encoded)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "model": model},
        "models/trained_model.pkl"
    )

    print("✅ Model saved successfully!")


if __name__ == "__main__":
    train_model()

