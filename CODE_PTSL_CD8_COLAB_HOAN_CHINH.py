# -*- coding: utf-8 -*-
# %%
# ================================================================
# PTSL - Banking Transaction Risk Analytics
# Colab-ready full code for Level 1 BI, Level 2 ML, Level 3 Anomaly/XAI
# ================================================================

import importlib
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTPUT_DIR = Path("ptsl_outputs")
FIG_DIR = OUTPUT_DIR / "figures"
STAR_SCHEMA_DIR = OUTPUT_DIR / "star_schema"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAR_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_package(import_name, pip_name=None):
    """Install a package in Colab if it is missing."""
    try:
        return importlib.import_module(import_name)
    except ImportError:
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return importlib.import_module(import_name)


def make_onehot_encoder():
    """Handle old/new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_current_figure(name):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {path}")


# %%
# ================================================================
# 1. Load source data
# ================================================================


def upload_if_needed(default_file="Banking_Transactional_Dataset.xlsx"):
    source_path = Path(default_file)
    if source_path.exists():
        return source_path

    print("Khong thay file du lieu trong Colab. Hay upload file Excel/CSV.")
    try:
        from google.colab import files

        uploaded = files.upload()
        for file_name in uploaded.keys():
            if file_name.lower().endswith((".xlsx", ".xls", ".csv")):
                return Path(file_name)
    except Exception as exc:
        raise FileNotFoundError(
            "Khong tim thay file du lieu. Hay upload Banking_Transactional_Dataset.xlsx."
        ) from exc

    raise FileNotFoundError("File upload khong phai Excel/CSV hop le.")


source_path = upload_if_needed()
print(f"Source file: {source_path}")

if source_path.suffix.lower() == ".csv":
    df_raw = pd.read_csv(source_path)
else:
    try:
        df_raw = pd.read_excel(source_path, sheet_name="Banking Data")
    except ValueError:
        df_raw = pd.read_excel(source_path)

print("Initial shape:", df_raw.shape)
display(df_raw.head())


# %%
# ================================================================
# 2. Data cleaning: duplicates, missing values, data types
# ================================================================

df = df_raw.copy()

print("===== DUPLICATE CHECK =====")
print("Full duplicate rows:", df.duplicated().sum())
if "TransactionID" in df.columns:
    print("Duplicate TransactionID:", df.duplicated(subset=["TransactionID"]).sum())
    df = df.drop_duplicates(keep="first")
    df = df.drop_duplicates(subset=["TransactionID"], keep="first")
else:
    df = df.drop_duplicates(keep="first")

print("Shape after duplicate handling:", df.shape)

print("\n===== MISSING VALUE CHECK =====")
missing_summary = pd.DataFrame(
    {
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
    }
).sort_values("missing_count", ascending=False)
display(missing_summary)

numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna("Unknown")

if "TransactionDate" in df.columns:
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

print("Total missing after handling:", df.isna().sum().sum())


# %%
# ================================================================
# 3. Outlier handling and feature engineering for BI
# ================================================================

required_cols = [
    "Amount",
    "LatePaymentAmount",
    "CreditCardFees",
    "InsuranceFees",
    "CustomerScore",
    "Channel",
]
missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

amount_cap_value = df["Amount"].quantile(0.99)
late_pay_cap_value = df["LatePaymentAmount"].quantile(0.99)

df["Amount_Cleaned"] = np.where(df["Amount"] > amount_cap_value, amount_cap_value, df["Amount"])
df["LatePayment_Cleaned"] = np.where(
    df["LatePaymentAmount"] > late_pay_cap_value,
    late_pay_cap_value,
    df["LatePaymentAmount"],
)

df["TotalFees"] = df["CreditCardFees"] + df["InsuranceFees"]

df["Fraud_Suspicion_Score"] = 0
df.loc[df["Amount"] > amount_cap_value, "Fraud_Suspicion_Score"] += 30
df.loc[df["LatePaymentAmount"] > late_pay_cap_value, "Fraud_Suspicion_Score"] += 30
df.loc[df["CustomerScore"] < 450, "Fraud_Suspicion_Score"] += 20

digital_high_amount = amount_cap_value * 0.70
df.loc[
    df["Channel"].isin(["Mobile", "Online"]) & (df["Amount"] > digital_high_amount),
    "Fraud_Suspicion_Score",
] += 20

df["Is_High_Risk"] = np.where(
    (df["Fraud_Suspicion_Score"] >= 60) | (df["CustomerScore"] < 400),
    1,
    0,
)

df["LatePayment_Flag"] = np.where(df["LatePaymentAmount"] > 0, 1, 0)
df["Risk_Status"] = np.where(df["Is_High_Risk"] == 1, "High Risk", "Normal")
df["Risk_Level"] = np.select(
    [
        df["Is_High_Risk"] == 1,
        df["Fraud_Suspicion_Score"] >= 40,
        df["Fraud_Suspicion_Score"] > 0,
    ],
    ["High", "Medium", "Low"],
    default="Normal",
)

df["Year"] = df["TransactionDate"].dt.year
df["Month"] = df["TransactionDate"].dt.month
df["MonthName"] = df["TransactionDate"].dt.strftime("%b")
df["Quarter"] = "Q" + df["TransactionDate"].dt.quarter.astype("Int64").astype(str)
df["YearMonth"] = df["TransactionDate"].dt.strftime("%Y-%m")
df["DateKey"] = df["TransactionDate"].dt.strftime("%Y%m%d").astype("Int64")

df["Income_Group"] = pd.cut(
    df["MonthlyIncome"],
    bins=[-np.inf, 3000, 7000, np.inf],
    labels=["Low Income", "Middle Income", "High Income"],
)

df["CreditScore_Group"] = pd.cut(
    df["CustomerScore"],
    bins=[-np.inf, 399, 549, 699, 850, np.inf],
    labels=["Very Low", "Low", "Medium", "Good", "Excellent"],
)

print("Amount cap value:", round(amount_cap_value, 2))
print("Late payment cap value:", round(late_pay_cap_value, 2))
print("High risk transactions:", int(df["Is_High_Risk"].sum()))
print("High risk rate:", f"{df['Is_High_Risk'].mean() * 100:.2f}%")
display(df.head())


# %%
# ================================================================
# 4. Export cleaned CSV for Power BI
# ================================================================

cleaned_csv = OUTPUT_DIR / "Cleaned_Banking_Data.csv"
df.to_csv(cleaned_csv, index=False, encoding="utf-8-sig")
print(f"Exported cleaned CSV: {cleaned_csv}")


# %%
# ================================================================
# 5. Star Schema exports for Power BI/Data Warehouse
# ================================================================

dim_date = (
    df[["DateKey", "TransactionDate", "Year", "Quarter", "Month", "MonthName", "YearMonth"]]
    .drop_duplicates()
    .sort_values("DateKey")
)

dim_channel = df[["Channel"]].drop_duplicates().sort_values("Channel").reset_index(drop=True)
dim_channel["ChannelKey"] = np.arange(1, len(dim_channel) + 1)

dim_transaction_type = (
    df[["TransactionType"]].drop_duplicates().sort_values("TransactionType").reset_index(drop=True)
)
dim_transaction_type["TransactionTypeKey"] = np.arange(1, len(dim_transaction_type) + 1)

dim_product = (
    df[["ProductCategory", "ProductSubcategory"]]
    .drop_duplicates()
    .sort_values(["ProductCategory", "ProductSubcategory"])
    .reset_index(drop=True)
)
dim_product["ProductKey"] = np.arange(1, len(dim_product) + 1)

dim_location = (
    df[["BranchCity", "BranchLat", "BranchLong"]]
    .drop_duplicates()
    .sort_values("BranchCity")
    .reset_index(drop=True)
)
dim_location["LocationKey"] = np.arange(1, len(dim_location) + 1)

dim_customer = (
    df.sort_values("TransactionDate")
    .drop_duplicates("CustomerID", keep="last")[
        [
            "CustomerID",
            "CustomerSegment",
            "CustomerScore",
            "CreditScore_Group",
            "MonthlyIncome",
            "Income_Group",
            "RecommendedOffer",
        ]
    ]
    .sort_values("CustomerID")
)

fact = df.merge(dim_channel, on="Channel", how="left")
fact = fact.merge(dim_transaction_type, on="TransactionType", how="left")
fact = fact.merge(dim_product, on=["ProductCategory", "ProductSubcategory"], how="left")
fact = fact.merge(dim_location, on=["BranchCity", "BranchLat", "BranchLong"], how="left")

fact_transactions = fact[
    [
        "TransactionID",
        "CustomerID",
        "DateKey",
        "ChannelKey",
        "TransactionTypeKey",
        "ProductKey",
        "LocationKey",
        "Amount",
        "Amount_Cleaned",
        "CreditCardFees",
        "InsuranceFees",
        "TotalFees",
        "LatePaymentAmount",
        "LatePayment_Cleaned",
        "LatePayment_Flag",
        "Fraud_Suspicion_Score",
        "Is_High_Risk",
        "Risk_Status",
        "Risk_Level",
    ]
]

star_tables = {
    "Fact_Transactions.csv": fact_transactions,
    "Dim_Date.csv": dim_date,
    "Dim_Channel.csv": dim_channel[["ChannelKey", "Channel"]],
    "Dim_TransactionType.csv": dim_transaction_type[["TransactionTypeKey", "TransactionType"]],
    "Dim_Product.csv": dim_product[["ProductKey", "ProductCategory", "ProductSubcategory"]],
    "Dim_Location.csv": dim_location[["LocationKey", "BranchCity", "BranchLat", "BranchLong"]],
    "Dim_Customer.csv": dim_customer,
}

for file_name, table in star_tables.items():
    table.to_csv(STAR_SCHEMA_DIR / file_name, index=False, encoding="utf-8-sig")

print("Exported Star Schema CSV files:")
for file_name in star_tables.keys():
    print("-", STAR_SCHEMA_DIR / file_name)


# %%
# ================================================================
# 6. KPI Summary for Level 1 BI
# ================================================================

total_transactions = len(df)
high_risk_transactions = int(df["Is_High_Risk"].sum())
late_payment_transactions = int(df["LatePayment_Flag"].sum())

kpi_summary = pd.DataFrame(
    {
        "KPI": [
            "Total Transactions",
            "High Risk Transactions",
            "High Risk Transaction Rate",
            "Late Payment Transactions",
            "Late Payment Ratio",
            "Average Risk Amount",
            "Average Fraud Suspicion Score",
            "Total Fee Revenue",
        ],
        "Value": [
            total_transactions,
            high_risk_transactions,
            high_risk_transactions / total_transactions,
            late_payment_transactions,
            late_payment_transactions / total_transactions,
            df.loc[df["Is_High_Risk"] == 1, "Amount"].mean(),
            df["Fraud_Suspicion_Score"].mean(),
            df["TotalFees"].sum(),
        ],
    }
)

display(kpi_summary)
kpi_summary.to_csv(OUTPUT_DIR / "KPI_Summary.csv", index=False, encoding="utf-8-sig")

channel_kpi = (
    df.groupby("Channel")
    .agg(
        Total_Transactions=("TransactionID", "count"),
        High_Risk_Transactions=("Is_High_Risk", "sum"),
        Channel_Risk_Ratio=("Is_High_Risk", "mean"),
        Late_Payment_Ratio=("LatePayment_Flag", "mean"),
        Avg_Fraud_Score=("Fraud_Suspicion_Score", "mean"),
        Total_Fees=("TotalFees", "sum"),
    )
    .reset_index()
    .sort_values("Channel_Risk_Ratio", ascending=False)
)
display(channel_kpi)
channel_kpi.to_csv(OUTPUT_DIR / "Channel_KPI.csv", index=False, encoding="utf-8-sig")


# %%
# ================================================================
# 7. EDA for Level 1 BI
# ================================================================

sns.set_theme(style="whitegrid")

# 7.1 Transaction trend and high-risk rate by month
monthly_eda = (
    df.groupby("YearMonth")
    .agg(
        Total_Transactions=("TransactionID", "count"),
        High_Risk_Rate=("Is_High_Risk", "mean"),
        Total_Amount=("Amount", "sum"),
        Total_Fees=("TotalFees", "sum"),
        Late_Payment_Ratio=("LatePayment_Flag", "mean"),
    )
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(14, 5))
sns.lineplot(data=monthly_eda, x="YearMonth", y="Total_Transactions", marker="o", ax=ax1)
ax1.set_title("Transaction Trend and High Risk Rate by Month")
ax1.set_xlabel("Year-Month")
ax1.set_ylabel("Total Transactions")
ax1.tick_params(axis="x", rotation=60)

ax2 = ax1.twinx()
sns.lineplot(
    data=monthly_eda,
    x="YearMonth",
    y="High_Risk_Rate",
    marker="o",
    color="crimson",
    ax=ax2,
)
ax2.set_ylabel("High Risk Rate")
save_current_figure("eda_monthly_transaction_risk_trend")

# 7.2 Channel risk comparison
plt.figure(figsize=(9, 5))
sns.barplot(data=channel_kpi, x="Channel", y="Channel_Risk_Ratio", palette="Set2")
plt.title("Channel Risk Ratio")
plt.ylabel("High Risk Transaction Rate")
save_current_figure("eda_channel_risk_ratio")

# 7.3 Customer segment risk
segment_kpi = (
    df.groupby("CustomerSegment")
    .agg(
        Total_Transactions=("TransactionID", "count"),
        High_Risk_Rate=("Is_High_Risk", "mean"),
        Avg_Amount=("Amount", "mean"),
        Avg_CustomerScore=("CustomerScore", "mean"),
        Late_Payment_Ratio=("LatePayment_Flag", "mean"),
    )
    .reset_index()
    .sort_values("High_Risk_Rate", ascending=False)
)
display(segment_kpi)

plt.figure(figsize=(10, 5))
sns.barplot(data=segment_kpi, x="CustomerSegment", y="High_Risk_Rate", palette="muted")
plt.title("High Risk Rate by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("High Risk Rate")
plt.xticks(rotation=20)
save_current_figure("eda_customer_segment_risk")

# 7.4 Amount by risk level
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Risk_Level", y="Amount_Cleaned", order=["Normal", "Low", "Medium", "High"])
plt.title("Transaction Amount by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Amount Cleaned")
save_current_figure("eda_amount_by_risk_level")

# 7.5 Customer score vs amount
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(min(4000, len(df)), random_state=RANDOM_STATE),
    x="CustomerScore",
    y="Amount_Cleaned",
    hue="Risk_Status",
    alpha=0.6,
)
plt.title("Customer Score vs Transaction Amount")
plt.xlabel("Customer Score")
plt.ylabel("Amount Cleaned")
save_current_figure("eda_customer_score_amount_risk")

# 7.6 Correlation heatmap
eda_numeric_cols = [
    "Amount",
    "Amount_Cleaned",
    "CustomerScore",
    "MonthlyIncome",
    "CreditCardFees",
    "InsuranceFees",
    "TotalFees",
    "LatePaymentAmount",
    "LatePayment_Cleaned",
    "Fraud_Suspicion_Score",
    "Is_High_Risk",
]
plt.figure(figsize=(11, 8))
sns.heatmap(df[eda_numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap for Transaction Risk Analytics")
save_current_figure("eda_correlation_heatmap")

# 7.7 Late payment monitoring
plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_eda, x="YearMonth", y="Late_Payment_Ratio", marker="o", color="darkorange")
plt.title("Late Payment Ratio by Month")
plt.xlabel("Year-Month")
plt.ylabel("Late Payment Ratio")
plt.xticks(rotation=60)
save_current_figure("eda_late_payment_ratio_by_month")


# %%
# ================================================================
# 8. Level 2 Machine Learning: classification models
# ================================================================

print(
    "Note: Is_High_Risk is rule-based and uses Amount, LatePaymentAmount, "
    "CustomerScore and Channel. Very high model scores are expected because "
    "the model can learn the target rules."
)

features_input = [
    "Amount",
    "TransactionType",
    "Channel",
    "ProductCategory",
    "CustomerScore",
    "MonthlyIncome",
    "CreditCardFees",
    "InsuranceFees",
    "LatePaymentAmount",
    "CustomerSegment",
]

X = df[features_input].copy()
y = df["Is_High_Risk"].copy()

numeric_features = [
    "Amount",
    "CustomerScore",
    "MonthlyIncome",
    "CreditCardFees",
    "InsuranceFees",
    "LatePaymentAmount",
]
categorical_features = ["TransactionType", "Channel", "ProductCategory", "CustomerSegment"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", make_onehot_encoder(), categorical_features),
    ],
    remainder="drop",
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

xgb_module = ensure_package("xgboost")
XGBClassifier = xgb_module.XGBClassifier

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    ),
}

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["liblinear", "lbfgs"],
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 8, 12],
        "classifier__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, 14],
    },
    "XGBoost": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.05, 0.1],
    },
}

trained_pipelines = {}
cv_scores_summary = {}

for name, model in models.items():
    print(f"\nTraining and tuning: {name}")
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=clf_pipeline,
        param_grid=param_grids[name],
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    trained_pipelines[name] = grid_search.best_estimator_
    cv_scores_summary[name] = grid_search.best_score_

    print("Best params:", grid_search.best_params_)
    print("Best CV F1:", round(grid_search.best_score_, 4))


# %%
# ================================================================
# 9. Model evaluation
# ================================================================

performance_metrics = []

for name, pipeline in trained_pipelines.items():
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    performance_metrics.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "CV_F1": cv_scores_summary[name],
        }
    )

df_results = pd.DataFrame(performance_metrics).sort_values(
    ["F1-score", "ROC-AUC"], ascending=False
)
display(df_results)
df_results.to_csv(OUTPUT_DIR / "Model_Performance.csv", index=False, encoding="utf-8-sig")

best_model_name = df_results.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
print("Best model:", best_model_name)

for name, pipeline in trained_pipelines.items():
    y_pred = pipeline.predict(X_test)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["Normal", "High Risk"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "High Risk"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    save_current_figure(f"ml_confusion_matrix_{name.replace(' ', '_').lower()}")

fig, ax = plt.subplots(figsize=(8, 6))
for name, pipeline in trained_pipelines.items():
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
plt.title("ROC Curve Comparison")
save_current_figure("ml_roc_curve_comparison")


# %%
# ================================================================
# 10. Feature importance and model interpretation
# ================================================================


def clean_feature_names(feature_names):
    return [name.replace("num__", "").replace("cat__", "") for name in feature_names]


feature_names = clean_feature_names(best_pipeline.named_steps["preprocessor"].get_feature_names_out())

tree_models = ["Decision Tree", "Random Forest", "XGBoost"]
fig, axes = plt.subplots(3, 1, figsize=(12, 16))

for i, name in enumerate(tree_models):
    pipeline = trained_pipelines[name]
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_
    names = clean_feature_names(pipeline.named_steps["preprocessor"].get_feature_names_out())

    order = np.argsort(importances)[::-1][:10]
    sns.barplot(ax=axes[i], x=importances[order], y=np.array(names)[order], palette="viridis")
    axes[i].set_title(f"Top Feature Importance - {name}")
    axes[i].set_xlabel("Importance")
    axes[i].set_ylabel("Feature")

save_current_figure("ml_tree_feature_importance")

lr_pipeline = trained_pipelines["Logistic Regression"]
lr_coef = lr_pipeline.named_steps["classifier"].coef_[0]
lr_names = clean_feature_names(lr_pipeline.named_steps["preprocessor"].get_feature_names_out())
order = np.argsort(np.abs(lr_coef))[::-1][:10]

plt.figure(figsize=(11, 6))
sns.barplot(x=lr_coef[order], y=np.array(lr_names)[order], palette="coolwarm")
plt.title("Top Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
save_current_figure("ml_logistic_regression_coefficients")

# Model-agnostic permutation importance on original columns
perm = permutation_importance(
    best_pipeline,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
perm_df = pd.DataFrame(
    {
        "Feature": X_test.columns,
        "Importance_Mean": perm.importances_mean,
        "Importance_STD": perm.importances_std,
    }
).sort_values("Importance_Mean", ascending=False)
display(perm_df)
perm_df.to_csv(OUTPUT_DIR / "Permutation_Importance.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
sns.barplot(data=perm_df, x="Importance_Mean", y="Feature", palette="mako")
plt.title(f"Permutation Importance - {best_model_name}")
save_current_figure("ml_permutation_importance")


# %%
# ================================================================
# 11. Level 3: Anomaly Detection
# ================================================================

anomaly_features = features_input
X_anomaly = df[anomaly_features].copy()

anomaly_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", make_onehot_encoder(), categorical_features),
    ],
    remainder="drop",
)
X_anomaly_ready = anomaly_preprocessor.fit_transform(X_anomaly)

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=RANDOM_STATE,
)
df["IF_Anomaly_Flag"] = np.where(isolation_forest.fit_predict(X_anomaly_ready) == -1, 1, 0)
df["IF_Anomaly_Score"] = -isolation_forest.decision_function(X_anomaly_ready)

lof = LocalOutlierFactor(n_neighbors=35, contamination=0.05)
df["LOF_Anomaly_Flag"] = np.where(lof.fit_predict(X_anomaly_ready) == -1, 1, 0)
df["LOF_Anomaly_Score"] = -lof.negative_outlier_factor_

df["Anomaly_Count"] = df["IF_Anomaly_Flag"] + df["LOF_Anomaly_Flag"]

df["Early_Warning_Score"] = (
    df["Is_High_Risk"] * 3
    + (df["Fraud_Suspicion_Score"] >= 60).astype(int) * 2
    + (df["Fraud_Suspicion_Score"].between(30, 59)).astype(int)
    + df["LatePayment_Flag"]
    + df["Anomaly_Count"]
)

df["Early_Warning_Level"] = pd.cut(
    df["Early_Warning_Score"],
    bins=[-1, 1, 3, 5, 10],
    labels=["Low", "Medium", "High", "Critical"],
)

anomaly_summary = pd.DataFrame(
    {
        "Method": ["Isolation Forest", "Local Outlier Factor", "Both Methods"],
        "Detected_Transactions": [
            int(df["IF_Anomaly_Flag"].sum()),
            int(df["LOF_Anomaly_Flag"].sum()),
            int((df["Anomaly_Count"] == 2).sum()),
        ],
        "Detected_Rate": [
            df["IF_Anomaly_Flag"].mean(),
            df["LOF_Anomaly_Flag"].mean(),
            (df["Anomaly_Count"] == 2).mean(),
        ],
    }
)
display(anomaly_summary)
anomaly_summary.to_csv(OUTPUT_DIR / "Anomaly_Summary.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 5))
anomaly_by_channel = (
    df.groupby("Channel")
    .agg(IF_Anomaly_Rate=("IF_Anomaly_Flag", "mean"), LOF_Anomaly_Rate=("LOF_Anomaly_Flag", "mean"))
    .reset_index()
    .melt(id_vars="Channel", var_name="Method", value_name="Anomaly_Rate")
)
sns.barplot(data=anomaly_by_channel, x="Channel", y="Anomaly_Rate", hue="Method")
plt.title("Anomaly Rate by Channel")
save_current_figure("anomaly_rate_by_channel")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(min(4000, len(df)), random_state=RANDOM_STATE),
    x="Amount_Cleaned",
    y="Fraud_Suspicion_Score",
    hue="Early_Warning_Level",
    alpha=0.65,
)
plt.title("Early Warning Level by Amount and Fraud Suspicion Score")
plt.xlabel("Amount Cleaned")
plt.ylabel("Fraud Suspicion Score")
save_current_figure("anomaly_early_warning_scatter")

anomaly_csv = OUTPUT_DIR / "Cleaned_Banking_Data_With_Anomaly.csv"
df.to_csv(anomaly_csv, index=False, encoding="utf-8-sig")
print(f"Exported anomaly-enhanced CSV: {anomaly_csv}")


# %%
# ================================================================
# 12. Level 3: Explainable AI with SHAP, fallback to permutation importance
# ================================================================

try:
    shap = ensure_package("shap")

    # SHAP works best here with a tree classifier on transformed data.
    shap_model_name = "XGBoost" if "XGBoost" in trained_pipelines else "Random Forest"
    shap_pipeline = trained_pipelines[shap_model_name]
    shap_model = shap_pipeline.named_steps["classifier"]
    shap_preprocessor = shap_pipeline.named_steps["preprocessor"]
    transformed_names = clean_feature_names(shap_preprocessor.get_feature_names_out())

    X_sample = X_test.sample(min(800, len(X_test)), random_state=RANDOM_STATE)
    X_sample_ready = shap_preprocessor.transform(X_sample)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_sample_ready)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(
        shap_values_to_plot,
        X_sample_ready,
        feature_names=transformed_names,
        show=False,
        max_display=12,
    )
    plt.title(f"SHAP Summary - {shap_model_name}")
    save_current_figure("xai_shap_summary")
    print("SHAP completed successfully.")
except Exception as exc:
    print("SHAP could not run in this environment.")
    print("Reason:", exc)
    print("Use Permutation_Importance.csv as the fallback explainability output.")


# %%
# ================================================================
# 13. Power BI DAX Measures to create manually
# ================================================================

dax_measures = r"""
Total Transactions =
COUNTROWS(Fact_Transactions)

High Risk Transactions =
CALCULATE(
    COUNTROWS(Fact_Transactions),
    Fact_Transactions[Is_High_Risk] = 1
)

High Risk Transaction Rate =
DIVIDE([High Risk Transactions], [Total Transactions])

Late Payment Transactions =
CALCULATE(
    COUNTROWS(Fact_Transactions),
    Fact_Transactions[LatePayment_Flag] = 1
)

Late Payment Ratio =
DIVIDE([Late Payment Transactions], [Total Transactions])

Average Risk Amount =
CALCULATE(
    AVERAGE(Fact_Transactions[Amount]),
    Fact_Transactions[Is_High_Risk] = 1
)

Average Fraud Suspicion Score =
AVERAGE(Fact_Transactions[Fraud_Suspicion_Score])

Channel Risk Ratio =
DIVIDE(
    CALCULATE(COUNTROWS(Fact_Transactions), Fact_Transactions[Is_High_Risk] = 1),
    COUNTROWS(Fact_Transactions)
)

Total Fee Revenue =
SUM(Fact_Transactions[TotalFees])

Total Late Payment Amount =
SUM(Fact_Transactions[LatePaymentAmount])
"""

dax_path = OUTPUT_DIR / "PowerBI_DAX_Measures.txt"
dax_path.write_text(dax_measures.strip(), encoding="utf-8")
print(f"Saved DAX measures: {dax_path}")
print(dax_measures)


# %%
# ================================================================
# 14. Final checklist
# ================================================================

print("""
DONE - Level 1 BI support:
- Cleaned CSV
- Outlier-handled fields
- Feature engineering
- Time fields for EDA/Power BI
- KPI summary
- Star Schema CSV exports
- EDA figures

DONE - Level 2 ML:
- Train/test split
- Encoding and scaling
- Cross validation
- Hyperparameter tuning
- Logistic Regression, Decision Tree, Random Forest, XGBoost
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix, ROC curve
- Feature importance and permutation importance

DONE - Level 3:
- Isolation Forest
- Local Outlier Factor
- Early warning score and level
- SHAP if available, permutation importance fallback

Power BI next:
1. Import CSV files in ptsl_outputs/star_schema.
2. Create relationships from dimensions to Fact_Transactions.
3. Add DAX measures from PowerBI_DAX_Measures.txt.
4. Build 5 dashboard pages:
   - Transaction Risk Overview
   - Abnormal Transactions
   - Customer Segment Risk
   - Channel Risk Analysis
   - Late Payment Monitoring
""")
