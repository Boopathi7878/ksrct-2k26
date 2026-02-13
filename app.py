from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ============================
# LOAD MODEL & SCALER
# ============================

model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    attack_count = 0
    normal_count = 0

    if request.method == "POST":

        file = request.files["file"]

        if file:
            df = pd.read_csv(file)

            if "Attack Type" in df.columns:
                df = df.drop("Attack Type", axis=1)

            X_scaled = scaler.transform(df)

            predictions = model.predict(X_scaled)
            predictions = np.where(predictions == -1, 1, 0)

            scores = model.decision_function(X_scaled)
            risk_score = (scores - scores.min()) / (scores.max() - scores.min())

            df["Prediction"] = predictions
            df["Risk Score"] = risk_score

            attack_count = int(sum(predictions))
            normal_count = len(predictions) - attack_count

            result = df.head().to_html(classes="table table-striped")

    return render_template(
        "index.html",
        result=result,
        attack_count=attack_count,
        normal_count=normal_count
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
