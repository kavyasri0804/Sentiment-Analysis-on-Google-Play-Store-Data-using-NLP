from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download once
nltk.download('punkt')
nltk.download('stopwords')

# Load model & vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("RandomForest_model.pkl", "rb"))

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

# ---------- HOME ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------- SENTIMENT ----------
@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    if request.method == "GET":
        return redirect(url_for("home"))

    review = request.form["review"]
    r = review.lower()

    if "average" in r:
        result = "Neutral"
    elif any(w in r for w in ["bad", "worst", "poor", "terrible"]):
        result = "Negative"
    elif any(w in r for w in ["good", "excellent", "nice", "great"]):
        result = "Positive"
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]

    return render_template("index.html", sentiment=result)

# ---------- ABOUT ----------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------- CONTACT ----------
@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run()
