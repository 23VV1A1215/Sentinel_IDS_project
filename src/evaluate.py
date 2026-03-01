from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
from preprocessing import preprocess


def evaluate_model(model_name):
    X, X_test_official, y, y_test_official = preprocess()

    model = joblib.load(f"models/{model_name}.pkl")

    y_pred = model.predict(X_test_official)

    print("\nClassification Report:")
    print(classification_report(y_test_official, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_official, y_pred))

    # ROC-AUC
    y_prob = model.predict_proba(X_test_official)[:, 1]
    auc = roc_auc_score(y_test_official, y_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test_official, y_prob)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.show()


if __name__ == "__main__":
    evaluate_model("XGBoost")