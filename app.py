from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            bedrooms = float(request.form["bedrooms"])
            bathrooms = float(request.form["bathrooms"])
            sqft = float(request.form["sqft"])

            features = np.array([[bedrooms, bathrooms, sqft]])
            prediction = model.predict(features)[0]

            return render_template("index.html", prediction=round(prediction, 2))
        except:
            return "Error in input. Please try again."
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
