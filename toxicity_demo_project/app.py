from flask import Flask, render_template, request, jsonify
import os
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)




MODEL_PATH = "./toxic_bert_model"

class ToxicModel:
    """
    Simple fallback so the demo still runs even if model.pkl doesn't exist yet.
    Replace with your real trained model from Google Colab.
    """
    toxic_words = {
        "idiot", "stupid", "hate", "trash", "moron", "loser",
        "ugly", "kill", "dumb", "garbage", "shut up"
    }

    labels = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]

    MODEL_PATH = "./toxic_bert_model"
    def  __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_PATH)

        self.model.eval()
    

    

    def detect_toxicity(self,text, threshold=0.5):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0]

        results = {
            label: float(prob)
            for label, prob in zip(self.labels, probs)
        }

        predicted = [
            label
            for label, prob in results.items()
            if prob >= threshold
        ]

        return predicted, results

    def predict(self, texts):
        results_bla = []
        if isinstance(texts, list):
            for text in texts:
                predicted, bla = self.detect_toxicity(text)
                results_bla.append(" ".join(predicted) if predicted else "")
            return results_bla
        else:

            predicted, results = self.detect_toxicity(texts)
        
        return predicted
    
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
        # try:
            return ToxicModel()
        # except Exception as e:
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

    prediction = " ".join(model.predict(text))
    return jsonify({
        "mode": "single",
        "inputText": text,
        "prediction": prediction,
        "label": str(prediction) if prediction else "Not Toxic"
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
        pred = len(pred) > 0
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
