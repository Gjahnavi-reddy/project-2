# Analytics Pipeline

Run in order:

```bash
python 01_eda.py
python 02_modeling.py
python reload_model.py
```

`01_eda.py` loads `sns.load_dataset("titanic")` once, saves `titanic.csv`, applies the assignment's missingness rules, produces EDA figures, and writes `eda_report.txt`.

`02_modeling.py` reads the committed `titanic.csv`, performs a stratified split before model preprocessing, trains Logistic Regression / Decision Tree / Random Forest, compares imbalance strategies, tunes Random Forest, performs the fare regression task, and saves the complete fitted classification pipeline with joblib.
