from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ModelClass import PredictionModel
import artifacts
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # allow all origins, adjust if needed

service = PredictionModel(artifacts_dir="artifacts")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    modelResult = service.predict(data, return_proba=True)
    
    title_map = {
        "confirmed": "Confirmed Exoplanet",
        "candidate": "Exoplanet Candidate",
        "false_positive": "False Positive"
    }

    description_map = {
        "confirmed": "Congratulations! The data strongly suggests a confirmed exoplanet detection. All parameters fall within expected ranges for a genuine planetary transit.",
        "candidate": "Promising signals detected. The data shows characteristics consistent with a planetary transit, but additional observations are recommended for confirmation.",
        "false_positive": "Analysis indicates this signal is likely caused by stellar activity, eclipsing binary stars, or instrumental effects rather than a genuine exoplanet."
    }

    result_type = modelResult["prediction"].lower().replace(' ', '_')
    result = {
        "type": result_type,
        "confidence": modelResult['probabilities'][modelResult['prediction']] * 100,
        "title": title_map[result_type],
        "description": description_map[result_type]
    }

    return jsonify(result)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/")
def home():
    return "Flask API is running!"
