from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":

        age = float(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Create DataFrame for pipeline
        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        # Add bmi_category (same as training)
        input_df["bmi_category"] = pd.cut(
            input_df["bmi"],
            bins=[0.0,18.5,25.0,29.9,float('inf')],
            labels=["underweight","nomralweight","overweight","obesity"]
        )

        # Transform input and predict
        input_prepared = pipeline.transform(input_df)
        prediction = model.predict(input_prepared)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
