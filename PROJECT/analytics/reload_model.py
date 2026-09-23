from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "best_classification_pipeline.joblib")

raw_example = pd.DataFrame([{
    "pclass": 3,
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 8.05,
    "sex": "male",
    "embarked": "S",
}])

print("Prediction from raw input:", model.predict(raw_example).tolist())
print("Probability:", model.predict_proba(raw_example).round(4).tolist())
