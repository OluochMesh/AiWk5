import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# 1. LOAD DATA
def load_data(path):
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    return df


# 2. CLEAN DATA
def clean_data(df):
    """Clean and prepare dataset."""

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop columns with mostly missing values (‘weight’ is useless)
    df.drop(columns=["weight"], inplace=True, errors='ignore')

    # Drop identifier column
    df.drop(columns=["patient_nbr"], inplace=True, errors='ignore')

    # Convert target variable into binary:
    #   <30  → 1 (readmitted within 30 days)
    #   NO, >30 → 0
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    return df


# 3. SPLIT FEATURES AND LABELS
def split_features_labels(df):
    """Separate features and target variable."""
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    return X, y


# 4. PREPROCESSING PIPELINE
def build_preprocessing_pipeline(X):
    """Build preprocessing pipeline for categorical + numeric columns."""

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ✅ Numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # ✅ Categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ✅ Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor



# 5. TRAIN/VAL/TEST SPLIT
def split_data(X, y):
    """Train/validation/test split with stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# 6. MAIN SCRIPT
def main():
    path = "./data/readmission.csv"

    print("✅ Loading dataset...")
    df = load_data(path)

    print("✅ Cleaning dataset...")
    df = clean_data(df)

    print("✅ Splitting features and labels...")
    X, y = split_features_labels(df)

    print("✅ Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(X)

    print("✅ Splitting into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("✅ Preprocessing done!")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # ✅ Optional: Save processed dataset (usually not needed)
    # X_train.to_csv("processed_X_train.csv", index=False)

if __name__ == "__main__":
    main()
