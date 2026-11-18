
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks"
]

def make_dataset():
    return pd.DataFrame({
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8],
        "attendance": [40, 50, 55, 65, 75, 85, 90, 95, 45, 60, 80, 92],
        "previous_marks": [20, 28, 35, 42, 60, 70, 80, 90, 25, 40, 68, 88],
        "passed": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    })

def train_model():
    df = make_dataset()

    X = df[FEATURES]
    y = df["passed"]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    model.fit(X, y)

    return model

def main():
    model = train_model()

    sample = make_dataset()[FEATURES].iloc[:2]

    predictions = model.predict(sample)

    print("Training completed successfully.")
    print("Sample predictions:", predictions.tolist())
    print("Prediction count:", len(predictions))

if __name__ == "__main__":
    main()
