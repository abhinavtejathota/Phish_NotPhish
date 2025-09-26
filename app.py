# app.py
from flask import Flask, request, jsonify, render_template_string
import joblib
import traceback
from feature_extract import extract_features_for_model
import json

MODEL_PATH = "robust_rf.pkl"
META_PATH = "robust_meta.json"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)
MODEL_FEATURES = meta["features"]

INDEX_HTML = """
<!doctype html>
<title>Phishing check (URL only)</title>
<h3>Paste URL and click Check</h3>
<form action="/predict" method="post">
  <input name="url" size="80" placeholder="http://example.com/path"><br><br>
  <input type="submit" value="Check">
</form>
<pre id="result">{{ result }}</pre>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, result="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            url = data.get("url", "").strip()
        else:
            url = request.form.get("url", "").strip()

        if not url:
            return jsonify({"error": "no url provided"}), 400

        X = extract_features_for_model(url, meta_path=META_PATH) 
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
        meaning = "phishing" if pred==1 else "legitimate"

        resp = {"url": url, "prediction": pred, "meaning": meaning}
        if proba is not None:
            resp["probabilities"] = proba
        if not request.is_json:
            return render_template_string(INDEX_HTML, result=json.dumps(resp, indent=2))
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": "exception", "detail": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
