import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score

# Load the encoded data
df_final = pd.read_csv("heart_disease_uci_encoded.csv")

# 1. DATA PREPARATION (Final Step)
# Create the binary target variable 'HeartDisease': 1 if num > 0, 0 otherwise.
df_final['HeartDisease'] = np.where(df_final['num'] > 0, 1, 0)
df_final = df_final.drop('num', axis=1) # Drop the original 'num' column

# 2. Define X and Y
X = df_final.drop('HeartDisease', axis=1)
Y = df_final['HeartDisease']

# 3. Train-Test Split (80/20, stratified to keep class balance)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# 4. Train the Optimized Model (Random Forest Classifier)
# Parameters are set for high performance and speed
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
final_model.fit(X_train, Y_train)

# 5. EVALUATE
Y_pred_final = final_model.predict(X_test)

# Calculate final key metrics (F1-Score and Recall are crucial)
accuracy_final = accuracy_score(Y_test, Y_pred_final)
f1_final = f1_score(Y_test, Y_pred_final)
recall_final = recall_score(Y_test, Y_pred_final)

# Extract Feature Importance (for Insights Analyst)
importance_final = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# --- Provide visible output so the script appears to run when executed ---
print(f"Accuracy: {accuracy_final:.4f}")
print(f"F1 score: {f1_final:.4f}")
print(f"Recall: {recall_final:.4f}")

print("\nTop features by importance:")
print(importance_final.head(10).to_string(index=False))

# Save feature importances and the trained model to disk for later use
importance_final.to_csv("feature_importances.csv", index=False)

import pickle
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("\nSaved feature_importances.csv and final_model.pkl")

# Save model metadata (feature order + label mapping) so predictions align with training
import json
metadata = {
    'feature_names': list(X.columns),
    'label_map': {"0": "No Heart Disease", "1": "Heart Disease"}
}
with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved model_metadata.json")
