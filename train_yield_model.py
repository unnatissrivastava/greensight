"""
Trains a real ML model to predict crop yield (in quintals/acre)
from rainfall, temperature, soil moisture, and fertilizer use.

NOTE ON DATA: We don't have access to a real farm-sensor dataset,
so this script generates a realistic SYNTHETIC dataset using a
known agronomic relationship + random noise. The model itself is
genuinely trained on this data with a real train/test split — the
predictions are real outputs of a trained model, not hardcoded.
Swap in a real CSV later by replacing the data-generation block.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

np.random.seed(42)
N = 2000

rainfall = np.random.uniform(300, 1500, N)          # mm per season
temperature = np.random.uniform(15, 40, N)           # deg C average
soil_moisture = np.random.uniform(10, 45, N)         # % volumetric
fertilizer = np.random.uniform(0, 200, N)            # kg/acre

# Realistic-ish relationship: yield rises with rainfall & moisture up to
# a point, dips in extreme heat, and responds to fertilizer with
# diminishing returns. Plus random noise so it isn't a perfect formula.
yield_qtl = (
    10
    + 0.02 * rainfall
    + 0.05 * soil_moisture
    + 0.08 * fertilizer
    - 0.00004 * fertilizer**2
    - 0.15 * (temperature - 27) ** 2
    + np.random.normal(0, 4, N)
)
yield_qtl = np.clip(yield_qtl, 2, None)

df = pd.DataFrame({
    "rainfall": rainfall,
    "temperature": temperature,
    "soil_moisture": soil_moisture,
    "fertilizer": fertilizer,
    "yield_qtl": yield_qtl
})

X = df[["rainfall", "temperature", "soil_moisture", "fertilizer"]]
y = df["yield_qtl"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R² score on test data: {r2_score(y_test, preds):.3f}")
print(f"Mean absolute error: {mean_absolute_error(y_test, preds):.2f} quintals/acre")

joblib.dump(model, "yield_model.pkl")
print("Saved trained model to yield_model.pkl")