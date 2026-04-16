from flask import Flask, render_template, request, jsonify
import os
import joblib

app = Flask(__name__)

MODEL_PATH = "model.pkl"

class FallbackToxicModel:
    """
    Simple fallback so the demo still runs even if model.pkl doesn't exist yet.
    Replace with your real trained model from Google Colab.
    """
    toxic_words = {
        "idiot", "stupid", "hate", "trash", "moron", "loser",
        "ugly", "kill", "dumb", "garbage", "shut up"
    }

    def predict(self, texts):
        results = []
        for text in texts:
            lowered = text.lower()
            is_toxic = 1 if any(word in lowered for word in self.toxic_words) else 0
            results.append(is_toxic)
        return results

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Could not load model.pkl, using fallback model instead: {e}")
    else:
        print("model.pkl not found, using fallback model.")
    return FallbackToxicModel()

model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = int(model.predict([text])[0])
    return jsonify({
        "mode": "single",
        "inputText": text,
        "prediction": prediction,
        "label": "Toxic" if prediction == 1 else "Not Toxic"
    })

@app.route("/analyze-comments", methods=["POST"])
def analyze_comments():
    data = request.get_json(force=True)
    comments = data.get("comments", [])

    if not isinstance(comments, list) or len(comments) == 0:
        return jsonify({"error": "comments must be a non-empty list"}), 400

    cleaned_comments = [str(c).strip() for c in comments if str(c).strip()]
    if not cleaned_comments:
        return jsonify({"error": "No valid comments after cleaning"}), 400

    predictions = model.predict(cleaned_comments)

    results = []
    toxic_count = 0

    for comment, pred in zip(cleaned_comments, predictions):
        pred = int(pred)
        if pred == 1:
            toxic_count += 1
        results.append({
            "comment": comment,
            "prediction": pred,
            "label": "Toxic" if pred == 1 else "Not Toxic"
        })

    total = len(cleaned_comments)
    toxicity_percent = round((toxic_count / total) * 100, 2)

    if toxicity_percent < 20:
        recommendation = "Looks relatively safe"
    elif toxicity_percent < 40:
        recommendation = "Use caution"
    else:
        recommendation = "High toxicity risk"

    return jsonify({
        "mode": "batch",
        "totalComments": total,
        "toxicComments": toxic_count,
        "toxicityPercent": toxicity_percent,
        "recommendation": recommendation,
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True)
