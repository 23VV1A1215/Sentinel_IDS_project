from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

from preprocessing import preprocess


def train_and_tune():
    X, X_test_official, y, y_test_official = preprocess()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTuning XGBoost...")

    param_grid = {
        "n_estimators": [200],
        "max_depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8]
    }

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print("\nBest Parameters:", grid.best_params_)
    print("Validation Accuracy:", acc)

    joblib.dump(best_model, "models/XGBoost_Tuned.pkl")

    return best_model


if __name__ == "__main__":
    train_and_tune()