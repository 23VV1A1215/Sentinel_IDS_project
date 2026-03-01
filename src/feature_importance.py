import matplotlib.pyplot as plt
import joblib
from preprocessing import preprocess

X, _, y, _ = preprocess()

model = joblib.load("models/XGBoost_Tuned.pkl")

importances = model.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()