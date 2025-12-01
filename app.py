# app.py
import os
import io
import base64
import sys

# Force non-interactive backend BEFORE importing pyplot
# Set env var and call matplotlib.use to ensure Agg backend (server-side PNG rendering)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Debug helper: print current working directory so you can confirm template path
print("Working dir:", os.getcwd(), file=sys.stderr)

CSV_PATH = "student_scores.csv"
MODEL_PATH = "student_score_model.joblib"

app = Flask(__name__)

# Utility: train model (used if model file not found or user asks to retrain)
def train_model(csv_path=CSV_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)
    X = df[["Hours"]].values
    y = df["Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)

    # Compute metrics on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    return model, {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "slope": model.coef_[0], "intercept": model.intercept_}

# Load or train model at startup
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        # If model file is corrupted or incompatible, retrain
        print("Failed to load model.joblib:", e, file=sys.stderr)
        model, metrics = train_model()
    else:
        # compute metrics from CSV for display
        try:
            df = pd.read_csv(CSV_PATH)
            X = df[["Hours"]].values
            y = df["Score"].values
            preds = model.predict(X)
            metrics = {
                "r2": float(r2_score(y, preds)),
                "mae": float(mean_absolute_error(y, preds)),
                "mse": float(mean_squared_error(y, preds)),
                "rmse": float(mean_squared_error(y, preds) ** 0.5),
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
            }
        except Exception as e:
            print("Error computing metrics:", e, file=sys.stderr)
            metrics = {}
else:
    model, metrics = train_model()

# Helper: generate plot as base64 PNG
def plot_regression_base64(model, csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    X = df[["Hours"]].values
    y = df["Score"].values

    # Sort for clean line
    sorted_idx = np.argsort(X[:, 0])
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]
    y_line = model.predict(X_sorted)

    # Create figure using Agg backend (no GUI)
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, label="Data")
    plt.plot(X_sorted, y_line, color="red", label="Regression line")
    plt.xlabel("Hours")
    plt.ylabel("Score")
    plt.title("Hours vs Score")
    plt.legend()
    plt.grid(alpha=0.2)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64

@app.route("/", methods=["GET", "POST"])
def index():
    global model, metrics
    result = None
    error = None
    input_text = ""
    # By default show plot and metrics
    plot_png = plot_regression_base64(model)

    if request.method == "POST":
        # Two buttons possible: Predict or Retrain
        if request.form.get("action") == "retrain":
            model, metrics = train_model()
            plot_png = plot_regression_base64(model)
            return redirect(url_for("index"))

        input_text = request.form.get("hours", "").strip()
        if not input_text:
            error = "Please enter hours (single number or comma-separated list)."
        else:
            try:
                # allow comma-separated multiple values
                parts = [p.strip() for p in input_text.split(",") if p.strip() != ""]
                hours = np.array([[float(p)] for p in parts])
                preds = model.predict(hours)
                # pair inputs with predictions
                result = [{"hours": float(h[0]), "predicted_score": float(round(p, 2))} for h, p in zip(hours.tolist(), preds.tolist())]
            except ValueError:
                error = "Invalid input. Use numbers like '9.25' or '9.25, 6.1'."
            except Exception as e:
                error = f"Prediction error: {e}"

    # refresh plot (in case model retrained)
    plot_png = plot_regression_base64(model)

    return render_template("index.html",
                           result=result,
                           error=error,
                           metrics=metrics,
                           plot_png=plot_png,
                           input_text=input_text)


@app.route("/predict", methods=["POST"])
def predict_api():
    global model
    data = request.get_json(force=True, silent=True)
    if not data or "hours" not in data:
        return jsonify({"error": "Missing 'hours' in JSON"}), 400

    try:
        hours_val = data["hours"]
        # support single number or list
        if isinstance(hours_val, list):
            hours = np.array([[float(h)] for h in hours_val])
            preds = model.predict(hours)
            return jsonify({"predictions": [float(round(p, 2)) for p in preds.tolist()]})
        else:
            hours = float(hours_val)
            pred = float(model.predict(np.array([[hours]]))[0])
            return jsonify({"predicted_score": float(round(pred, 2))})
    except ValueError:
        return jsonify({"error": "'hours' must be a number or list of numbers."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Development server only — set debug=False and use_reloader=False to avoid
    # starting extra threads/processes that can conflict with Matplotlib/Tk.
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
