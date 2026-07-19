"""
Trains a real ML classifier to label a leaf photo as
"Healthy", "Mild stress", or "High stress" from color-based
features extracted from the image (healthy-pixel %, stressed-pixel %,
average RGB, and RGB variance as a texture proxy).

NOTE ON DATA: We don't have access to a labeled real leaf-disease
photo dataset, so this script generates a realistic SYNTHETIC feature
dataset (the same honest approach used in train_yield_model.py):
each class has a characteristic but overlapping distribution over the
features, plus random noise, so the classes aren't perfectly separable.
The classifier is genuinely trained on this data with a real
train/test split, and its predictions on new feature vectors are real
model outputs, not hardcoded rules.

Swap in real labeled photos later by replacing the data-generation
block with actual extracted features from a labeled image folder.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

np.random.seed(42)

N_PER_CLASS = 700
rows = []

# Healthy: mostly green pixels, low stress pixels, fairly uniform texture
for _ in range(N_PER_CLASS):
    healthy_pct = np.clip(np.random.normal(78, 12), 0, 100)
    stressed_pct = np.clip(np.random.normal(3, 3), 0, 100)
    mean_g = np.random.normal(150, 15)
    mean_r = mean_g - np.random.normal(35, 10)
    mean_b = np.random.normal(70, 12)
    texture_std = np.random.normal(18, 5)
    rows.append([healthy_pct, stressed_pct, mean_r, mean_g, mean_b, texture_std, "Healthy"])

# Mild stress: mixed green/yellow, moderate stress pixels, more texture variance
for _ in range(N_PER_CLASS):
    healthy_pct = np.clip(np.random.normal(50, 15), 0, 100)
    stressed_pct = np.clip(np.random.normal(18, 8), 0, 100)
    mean_g = np.random.normal(135, 15)
    mean_r = mean_g - np.random.normal(15, 10)
    mean_b = np.random.normal(65, 12)
    texture_std = np.random.normal(28, 7)
    rows.append([healthy_pct, stressed_pct, mean_r, mean_g, mean_b, texture_std, "Mild stress"])

# High stress: browning/yellowing dominant, high stress pixels, high texture variance
for _ in range(N_PER_CLASS):
    healthy_pct = np.clip(np.random.normal(20, 12), 0, 100)
    stressed_pct = np.clip(np.random.normal(45, 15), 0, 100)
    mean_g = np.random.normal(110, 18)
    mean_r = mean_g + np.random.normal(10, 12)
    mean_b = np.random.normal(55, 12)
    texture_std = np.random.normal(38, 9)
    rows.append([healthy_pct, stressed_pct, mean_r, mean_g, mean_b, texture_std, "High stress"])

df = pd.DataFrame(
    rows,
    columns=["healthy_pct", "stressed_pct", "mean_r", "mean_g", "mean_b", "texture_std", "label"],
)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[["healthy_pct", "stressed_pct", "mean_r", "mean_g", "mean_b", "texture_std"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Accuracy on test data: {accuracy_score(y_test, preds):.3f}")
print("\nClassification report:")
print(classification_report(y_test, preds))
print("Confusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, preds, labels=model.classes_),
    index=model.classes_, columns=model.classes_
))

joblib.dump(model, "disease_model.pkl")
print("\nSaved trained model to disease_model.pkl")