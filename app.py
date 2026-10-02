from flask import Flask, render_template, request
import joblib


# Create Flask application
app = Flask(__name__)


# Load trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get news text from the form
    news_text = request.form["news"]

    # Convert the text using the trained TF-IDF vectorizer
    news_tfidf = vectorizer.transform([news_text])

    # Predict FAKE or REAL
    prediction = model.predict(news_tfidf)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(news_tfidf)[0]

    # Find confidence of predicted class
    class_names = model.classes_
    predicted_index = list(class_names).index(prediction)

    confidence = probabilities[predicted_index] * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)