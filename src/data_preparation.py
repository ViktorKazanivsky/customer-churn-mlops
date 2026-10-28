import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()

    # ID клієнта не несе корисної інформації для моделі
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges містить пробіли замість чисел -> перетворюємо в число
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"No": 0, "Yes": 1})

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    return X, y.astype(int)