from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "titanic.csv")

# Modeling target/features. The saved pipeline accepts raw rows with these columns.
target = "survived"
clf_features = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
X = df[clf_features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric = ["pclass", "age", "sibsp", "parch", "fare"]
categorical = ["sex", "embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, oob_score=True
    ),
}

rows = []
roc_data = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob),
    })
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_data[name] = (fpr, tpr)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm)
    ax.set_title(f"{name} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG / f"{name.lower().replace(' ', '_')}_confusion.png", dpi=150)
    plt.close(fig)

# Decision tree visualization.
tree_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
])
tree_pipe.fit(X_train, y_train)
feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["not_survived", "survived"],
    filled=False,
    max_depth=3,
    ax=ax,
)
fig.tight_layout()
fig.savefig(FIG / "decision_tree.png", dpi=150)
plt.close(fig)

# ROC comparison.
fig, ax = plt.subplots(figsize=(7, 5))
for name, (fpr, tpr) in roc_data.items():
    ax.plot(fpr, tpr, label=name)
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC comparison")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "roc_comparison.png", dpi=150)
plt.close(fig)

comparison = pd.DataFrame(rows)

# Imbalance comparison using Random Forest.
imbalance = []
variants = {
    "baseline": RandomForestClassifier(n_estimators=300, random_state=42),
    "class_weight_balanced": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
}
for label, estimator in variants.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    imbalance.append({
        "variant": label,
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })

smote_pipe = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
])
smote_pipe.fit(X_train, y_train)
smote_pred = smote_pipe.predict(X_test)
imbalance.append({
    "variant": "SMOTE_train_only",
    "precision": precision_score(y_test, smote_pred),
    "recall": recall_score(y_test, smote_pred),
    "f1": f1_score(y_test, smote_pred),
})
imbalance_df = pd.DataFrame(imbalance)

# Grid search for Random Forest. OOB score is calculated by refitting the best
# hyperparameters with oob_score=True after CV.
rf_pipe = Pipeline([("preprocess", preprocessor), (
    "model", RandomForestClassifier(random_state=42, oob_score=True)
)])
grid = GridSearchCV(
    rf_pipe,
    {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best_params = grid.best_params_
best_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=best_params["model__n_estimators"],
        max_depth=best_params["model__max_depth"],
        max_features=best_params["model__max_features"],
        random_state=42,
        oob_score=True,
    )),
])
best_rf.fit(X_train, y_train)

# Regression: predict fare from other available features.
reg_df = df.drop(columns=["fare"]).copy()
reg_target = df["fare"].copy()
reg_features = ["pclass", "age", "sibsp", "parch", "survived", "sex", "embarked"]

Xr = reg_df[reg_features]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, reg_target, test_size=0.20, random_state=42
)

reg_num = ["pclass", "age", "sibsp", "parch", "survived"]
reg_cat = ["sex", "embarked"]
reg_pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), reg_cat),
])
reg_pipe = Pipeline([
    ("preprocess", reg_pre),
    ("model", LinearRegression()),
])
reg_pipe.fit(Xr_train, yr_train)
reg_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test, reg_pred)
rmse = mean_squared_error(yr_test, reg_pred, squared=False)
r2 = r2_score(yr_test, reg_pred)
n = len(yr_test)
p = reg_pipe.named_steps["preprocess"].transform(Xr_test).shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

residuals = yr_test - reg_pred
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(reg_pred, residuals, alpha=0.6)
ax.axhline(0, linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual")
ax.set_title("Fare regression residual plot")
fig.tight_layout()
fig.savefig(FIG / "fare_residuals.png", dpi=150)
plt.close(fig)

# Choose the classification model for deployment by F1 on the held-out test set.
# This is a documented metric-based selection, not a claim about universal superiority.
selected_name = comparison.sort_values("f1", ascending=False).iloc[0]["model"]
selected_estimator = models[selected_name]
final_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", selected_estimator),
])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, ROOT / "best_classification_pipeline.joblib")

with (ROOT / "model_report.txt").open("w", encoding="utf-8") as f:
    f.write("CLASSIFICATION COMPARISON\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\nCLASS BALANCE\n")
    f.write(y.value_counts(normalize=True).rename("proportion").to_string())
    f.write("\n\nIMBALANCE COMPARISON\n")
    f.write(imbalance_df.to_string(index=False))
    f.write("\n\nGRID SEARCH\n")
    f.write(str(best_params) + "\n")
    f.write(f"OOB score={best_rf.named_steps['model'].oob_score_:.6f}\n")
    f.write("\nREGRESSION\n")
    f.write(f"MAE={mae:.6f}\nRMSE={rmse:.6f}\nR2={r2:.6f}\nAdjusted_R2={adj_r2:.6f}\n")
    f.write("\nHETEROSCEDASTICITY NOTE\n")
    f.write("Inspect figures/fare_residuals.png. A visibly changing residual spread across fitted values "
            "should be reported as evidence of heteroscedasticity; a roughly constant random spread should "
            "be reported as no clear visual evidence of heteroscedasticity.\n")
    f.write(f"\nSELECTED CLASSIFIER BY TEST F1: {selected_name}\n")
    f.write("Selection is based on the held-out F1 value and should be explained using the actual values generated by this run.\n")

print("Modeling complete.")
print(comparison.to_string(index=False))
