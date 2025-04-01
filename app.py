from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# ✅ Load the trained model & vectorizer
try:
    model = joblib.load("URL3.pickle")  # Ensure this is the correct model file
    cv = joblib.load("count_vectorizer_2.pickle")  # Ensure this is the correct vectorizer file
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model, cv = None, None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        URL = request.form.get("URL", "").strip()  # Get user input

        if not URL:
            return render_template("index.html", prediction="⚠️ Please enter a URL.")

        # ✅ Ensure model & vectorizer are loaded
        if model is None or cv is None:
            return render_template("index.html", prediction="⚠️ Model not loaded correctly.")

        try:
            # ✅ Convert input to a feature vector using the loaded vectorizer
            input_features = cv.transform([URL])  # 🔥 This is the FIXED line!

            # ✅ Predict using the trained model
            pred = model.predict(input_features)[0]
            prediction = "✅ Real URL" if pred == 1 else "❌ Fake URL"
        except Exception as e:
            prediction = f"⚠️ Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
