"""
Trains the crop yield model on a REAL dataset (109 real records) instead
of the synthetic data used before.

Source: Sukhman Singh's Crop Yield Prediction dataset
(https://github.com/SUKHMAN-SINGH-1612/Data-Science-Projects), saved
locally as data/crop_yield_real_data.xlsx.

Columns: Rain Fall (mm), Fertilizer (kg), Temperature (C),
Nitrogen (N), Phosphorus (P), Potassium (K) -> Yield (Q/acre)

NOTE ON DATA SIZE: this is a small real dataset (109 rows, ~99 after
cleaning). That's enough to train a genuine model and get an honest
R² score, but not enough to expect production-grade accuracy or wide
generalization. Treat this model as a real but modest baseline --
more real records would meaningfully improve it.

NOTE ON SCHEMA CHANGE: this real dataset has no soil-moisture column.
Instead of faking soil moisture, we switched the model's inputs to
match what the real data actually provides: rainfall, temperature,
fertilizer, and N/P/K macronutrient levels. app.py and the
yield-predictor form were updated to match.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_excel("data/crop_yield_real_data.xlsx")

df.columns = [
    "rainfall", "fertilizer", "temperature",
    "nitrogen", "phosphorus", "potassium", "yield_qtl",
]

# Clean: the raw sheet has ":" typos in temperature, and some blank cells
df["temperature"] = df["temperature"].replace(":", pd.NA)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where the target itself is missing -- can't train on those
df = df.dropna(subset=["yield_qtl"])

# Fill missing feature values with the column median (documented, not hidden)
for col in ["rainfall", "fertilizer", "temperature", "nitrogen", "phosphorus", "potassium"]:
    df[col] = df[col].fillna(df[col].median())

print(f"Training on {len(df)} real records after cleaning.")

feature_cols = ["rainfall", "temperature", "fertilizer", "nitrogen", "phosphorus", "potassium"]
X = df[feature_cols]
y = df["yield_qtl"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R² score on test data: {r2_score(y_test, preds):.3f}")
print(f"Mean absolute error: {mean_absolute_error(y_test, preds):.2f} quintals/acre")
print(f"(test set size: {len(y_test)} records -- small, so treat this score as indicative, not definitive)")

joblib.dump(model, "yield_model.pkl")
print("Saved trained model to yield_model.pkl")