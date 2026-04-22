import pandas as pd
import numpy as np
from scipy import stats
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

# Merge population data to get population for each row (for potential use as a feature)
population_data = pd.read_csv("./population_data/population_long_format.csv")
df = df.merge(population_data, on=["State", "Year", "Sex"], how="left")
print("Shape after merging with population data:", df.shape)

# Merge unemployment data to get unemployment rate for each row (for potential use as a feature)
unemployment_data = pd.read_csv("./unemployment_data/unemployment_long_format.csv")
df = df.merge(unemployment_data, on=["State", "Year", "Sex"], how="left")
print("Shape after merging with unemployment data:", df.shape)

# Integrated dataset information
print("\n── Integrated Dataset Info ──")
print("Unique Offences:", df["Offence"].nunique())
print("Unique States:", df["State"].nunique())
print("\nShape after merging datasets:", df.shape)
print("\nFirst 10 rows of the integrated dataset:")
print(df.head(10).to_string(index=False))

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
required_cols = ["Population", "Unemployment_Rate", "Rate_lag1", "Rate_lag2", "Rate_lag3", "Rate_roll3", 
                 "Rate_next", "State_mean_rate_lag1", "Rate_trend1", "Rate_trend2", "Rate_pct_chg1"]
df = df.dropna(subset=required_cols)
print("\nShape after feature engineering:", df.shape)

print("Last years in dataset:", sorted(df["Year"].unique())[-5:])

# ── 4. ENCODE CATEGORICALS ───────────────────────────────────────────────────
encoder = OneHotEncoder(sparse_output=False, drop="first")
cat_encoded = encoder.fit_transform(df[["Offence", "State", "Sex"]])
cat_cols    = encoder.get_feature_names_out(["Offence", "State", "Sex"])

features_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

NUMERIC_FEATURES = ["Year", "Population", "Unemployment_Rate", "Rate_lag1", "Rate_lag2", "Rate_lag3", "Rate_roll3", 
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
errors = {}  # To store prediction errors for analysis
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).clip(min=0)
    abs_errors = np.abs(y_test.values - y_pred)
 
    results[name] = {
        "MAE" : mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²"  : r2_score(y_test, y_pred),
    }
    errors[name] = abs_errors
    trained_models[name] = model
 
# ── 7. COMPARE PERFORMANCE ───────────────────────────────────────────────────
print("\n── Model Comparison ──")
print(f"{'Metric':<8}  {'Ridge':>10}  {'RandomForest':>14}")
print("-" * 52)
 
for metric in ["MAE", "RMSE", "R²"]:
    ridge_val = results["Ridge"][metric]
    rf_val    = results["RandomForest"][metric]
 
    print(f"{metric:<8}  {ridge_val:>10.4f}  {rf_val:>14.4f}")

# Statistical significance testing with paired t-test

ridge_errors = errors["Ridge"]
rf_errors    = errors["RandomForest"]
diff         = ridge_errors - rf_errors   # positive = Ridge was worse on that row
 
print("\n── Statistical Significance: Ridge vs RandomForest ──")
print(f"{'':30}  {'Statistic':>10}  {'p-value':>10}  {'Significant':>12}")
print("-" * 68)

# Assumes: differences are normally distributed (reasonable for large n)
# H0: mean absolute error is equal for both models
# H1: mean absolute error differs between models
t_stat, t_pval = stats.ttest_rel(ridge_errors, rf_errors)
t_sig = "Yes (p<0.05)" if t_pval < 0.05 else "No"
print(f"{'Paired t-test (MAE)':<30}  {t_stat:>10.4f}  {t_pval:>10.4f}  {t_sig:>12}")

# ── Summary interpretation ────────────────────────────────────────────────────
rf_wins    = np.sum(rf_errors < ridge_errors)   # RF had smaller error
ridge_wins = np.sum(ridge_errors < rf_errors)
ties       = np.sum(ridge_errors == rf_errors)
n_no_ties  = len(diff) - ties

print()
print("── Interpretation ──")
print(f"  Test rows          : {len(ridge_errors)}")
print(f"  RF wins (lower err): {rf_wins} / {n_no_ties} non-tied rows  ({100*rf_wins/n_no_ties:.1f}%)")
print(f"  Ridge wins         : {ridge_wins} / {n_no_ties} non-tied rows  ({100*ridge_wins/n_no_ties:.1f}%)")
print(f"  Mean |error| Ridge : {ridge_errors.mean():.4f}")
print(f"  Mean |error| RF    : {rf_errors.mean():.4f}")
print(f"  Mean diff (R-RF)   : {diff.mean():.4f}  (positive = Ridge worse on average)")
print()

# Interpretation
if t_pval < 0.05 and rf_errors.mean() < ridge_errors.mean():
    print("Verdict: RandomForest is significantly better than Ridge at predicting next year's crime rate.")
elif t_pval < 0.05 and ridge_errors.mean() < rf_errors.mean():
    print("Verdict: Ridge is significantly better than RandomForest at predicting next year's crime rate.")
else:
    print("Verdict: No statistically significant difference between Ridge and RandomForest in predicting next year's crime rate.")
 
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
 
print("\n── Sample Future Predictions ──")
print(
    latest[["Offence", "State", "Sex", "Year", "Rate", "Rate_next", "Ridge_Pred", "RF_Pred"]]
    .head(12)
    .to_string(index=False)
)

# Add this after predictions to find biggest RF errors
test_df = df[test_mask].copy()
test_df["RF_pred"] = trained_models["RandomForest"].predict(X_test)
test_df["error"]   = abs(test_df["RF_pred"] - y_test)

print("\n── Top 10 RF Prediction Errors ──")
print(test_df.nlargest(10, "error")[
    ["Offence", "State", "Sex", "Year", "Rate", "Rate_next", "RF_pred", "error"]
])