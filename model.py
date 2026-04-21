import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load preprocessed data
df = pd.read_csv("./preprocessed_data.csv")
print("Shape:", df.shape)
print(df.head(5))

# Drop unknown sexes
df = df[df["Sex"] != "Unknown"]

# Sort by group for feature engineering
df = df.sort_values(["Offence", "State", "Sex", "Year"]).reset_index(drop=True)

# Feature engineering: create lag and rolling features
group_cols = ["Offence", "State", "Sex"]

df["Rate_lag1"]  = df.groupby(group_cols)["Rate"].shift(1)
df["Rate_lag2"]  = df.groupby(group_cols)["Rate"].shift(2)
df["Rate_lag3"]  = df.groupby(group_cols)["Rate"].shift(3)
df["Rate_roll3"] = (
    df.groupby(group_cols)["Rate"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

# Year-on-year absolute change: positive = rising, negative = falling
df["Rate_trend1"]    = df.groupby(group_cols)["Rate"].transform(lambda x: x.diff().shift(1))
 
# 2-year trend: smooths out single-year noise
df["Rate_trend2"]    = df.groupby(group_cols)["Rate"].transform(lambda x: x.diff(2).shift(1))

# Percentage change: normalised momentum (handles large vs small rate groups)
df["Rate_pct_chg1"]  = df.groupby(group_cols)["Rate"].transform(
    lambda x: x.pct_change().shift(1).replace([np.inf, -np.inf], np.nan)
)

# State-year mean rate across all offences (captures overall state crime level)
state_year_mean = (
    df.groupby(["State", "Year"])["Rate"]
    .mean()
    .rename("State_mean_rate")
    .reset_index()
)
df = df.merge(state_year_mean, on=["State", "Year"], how="left")
df["State_mean_rate_lag1"] = df.groupby(group_cols)["State_mean_rate"].shift(1)

# Target: next year's rate
df["Rate_next"] = df.groupby(group_cols)["Rate"].shift(-1)

# Drop rows where lags or target are missing
required_cols = ["Rate_lag1", "Rate_lag2", "Rate_lag3", "Rate_roll3", 
                 "Rate_next", "State_mean_rate_lag1", "Rate_trend1", "Rate_trend2", "Rate_pct_chg1"]
df = df.dropna(subset=required_cols)
print("\nShape after feature engineering:", df.shape)

# ── 4. ENCODE CATEGORICALS ───────────────────────────────────────────────────
encoder = OneHotEncoder(sparse_output=False, drop="first")
cat_encoded = encoder.fit_transform(df[["Offence", "State", "Sex"]])
cat_cols    = encoder.get_feature_names_out(["Offence", "State", "Sex"])

features_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

NUMERIC_FEATURES = ["Year", "Rate_lag1", "Rate_lag2", "Rate_lag3", "Rate_roll3", 
                    "State_mean_rate_lag1", "Rate_trend1", "Rate_trend2", "Rate_pct_chg1"]
X = pd.concat([df[NUMERIC_FEATURES], features_df], axis=1)
y = df["Rate_next"]

# ── 5. TRAIN / TEST SPLIT (time-based) ───────────────────────────────────────
train_mask = df["Year"] <= 2021
test_mask  = df["Year"] >  2021

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")

# ── 6. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,    
        max_depth=10,        
        min_samples_leaf=5,  
        random_state=42,
        n_jobs=-1           
    ),
}
 
results = {} 
trained_models = {}  
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    results[name] = {
        "MAE" : mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²"  : r2_score(y_test, y_pred),
    }
    trained_models[name] = model
 
# ── 7. COMPARE PERFORMANCE ───────────────────────────────────────────────────
print("\n── Model Comparison ──")
print(f"{'Metric':<8}  {'Ridge':>10}  {'RandomForest':>14}  {'Winner':>12}")
print("-" * 52)
 
for metric in ["MAE", "RMSE", "R²"]:
    ridge_val = results["Ridge"][metric]
    rf_val    = results["RandomForest"][metric]
 
    # Lower is better for MAE/RMSE; higher is better for R²
    if metric == "R²":
        winner = "RandomForest" if rf_val > ridge_val else "Ridge"
    else:
        winner = "RandomForest" if rf_val < ridge_val else "Ridge"
 
    print(f"{metric:<8}  {ridge_val:>10.4f}  {rf_val:>14.4f}  {winner:>12}")
 
# ── 8. FEATURE IMPORTANCE (RF only) ──────────────────────────────────────────
print("\n── Top 10 Feature Importances (RandomForest) ──")
importances = pd.Series(
    trained_models["RandomForest"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)
 
print(importances.head(10).round(4).to_string())
 
# ── 9. FUTURE PREDICTIONS (both models) ──────────────────────────────────────
latest = (
    df.sort_values("Year")
    .groupby(group_cols)
    .last()
    .reset_index()
)
 
cat_latest  = encoder.transform(latest[["Offence", "State", "Sex"]])
feat_latest = pd.DataFrame(cat_latest, columns=cat_cols)
X_future    = pd.concat([latest[NUMERIC_FEATURES].reset_index(drop=True), feat_latest], axis=1)
 
latest["Ridge_Pred"] = trained_models["Ridge"].predict(X_future).round(2)
latest["RF_Pred"]    = trained_models["RandomForest"].predict(X_future).round(2)
latest["Pred_Diff"]  = (latest["RF_Pred"] - latest["Ridge_Pred"]).round(2)
 
print("\n── Sample Future Predictions (both models) ──")
print(
    latest[["Offence", "State", "Sex", "Year", "Rate", "Ridge_Pred", "RF_Pred", "Pred_Diff"]]
    .head(12)
    .to_string(index=False)
)

# Add this after predictions to find biggest RF errors
test_df = df[test_mask].copy()
test_df["RF_pred"] = trained_models["RandomForest"].predict(X_test)
test_df["error"]   = abs(test_df["RF_pred"] - y_test)

print(test_df.nlargest(10, "error")[
    ["Offence", "State", "Sex", "Year", "Rate", "Rate_next", "RF_pred", "error"]
])