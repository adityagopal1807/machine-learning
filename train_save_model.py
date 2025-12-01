# train_save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

CSV_PATH = "student_scores.csv"
MODEL_PATH = "student_score_model.joblib"

def train_and_save(csv_path=CSV_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)
    X = df[["Hours"]].values
    y = df["Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Metrics on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Trained LinearRegression. Slope(m)={model.coef_[0]:.4f}, Intercept(b)={model.intercept_:.4f}")
    print(f"Test metrics -> R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Save model
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_and_save()
