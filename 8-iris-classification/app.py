from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

saved = joblib.load("iris_classifier.pkl")

model = saved["model"]
feature_names = saved["feature_names"]
target_names = saved["target_names"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sample = [
            [
                float(data["sepal_length"]),
                float(data["sepal_width"]),
                float(data["petal_length"]),
                float(data["petal_width"]),
            ]
        ]

        prediction = model.predict(sample)[0]

        return jsonify(
            {"prediction": target_names[prediction], "class": int(prediction)}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
