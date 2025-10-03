from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from ModelClass import PredictionModel
import tempfile
import artifacts
import os
import pandas as pd
import io
from datetime import datetime

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


@app.route("/api/analyze_csv", methods=["POST"])
def analyze_csv():
    """
    Endpoint to process CSV data with ML model.
    
    Expected JSON format (array of objects):
    [
        {"orbper": 365.25, "trandep": 100, "trandur": 3.5, ...},
        {"orbper": 400.0, "trandep": 120, "trandur": 4.0, ...},
        ...
    ]
    
    Returns: JSON array with added prediction columns
    """
    try:
        # Get JSON data from request (array of objects)
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not isinstance(data, list):
            return jsonify({"error": "Data must be an array of objects"}), 400
        
        if len(data) == 0:
            return jsonify({"error": "Data array is empty"}), 400
        
        # Validate required fields
        required_fields = ['orbper', 'trandep', 'trandur', 'rade', 'insol', 
                          'eqt', 'teff', 'logg', 'rad']
        
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                return jsonify({
                    "error": f"Row {idx} must be an object/dictionary"
                }), 400
                
            missing_fields = [field for field in required_fields if field not in row]
            if missing_fields:
                return jsonify({
                    "error": f"Row {idx} missing required fields: {missing_fields}"
                }), 400
        
        # Process each row with the model
        results = []
        for idx, row in enumerate(data):
            try:
                prediction = service.predict(row, return_proba=True)
                # Add original data plus predictions
                result_row = row.copy()
                result_row['model_prediction'] = prediction['prediction'].lower().replace(' ', '_')
                result_row['confidence_score'] = prediction['probabilities'][prediction['prediction']] * 100
                results.append(result_row)
                
            except Exception as e:
                return jsonify({
                    "error": f"Error processing row {idx}: {str(e)}"
                }), 500
        
        # Return the array of objects with added columns
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Flask API is running!"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
