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
    
    Expected JSON format:
    [
        {"orbper": 365.25, "trandep": 100, "trandur": 3.5, ...},
        {"orbper": 400.0, "trandep": 120, "trandur": 4.0, ...},
        ...
    ]
    """
    print("Received request for /api/analyze_csv")
   
    try:
        # Get JSON data from request
        data = request.get_json()
        print(f"Data received: {data}")
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not isinstance(data, list):
            return jsonify({"error": "Data must be a list of objects"}), 400
        
        # Validate required fields
        required_fields = ['orbper', 'trandep', 'trandur', 'rade', 'insol', 
                          'eqt', 'teff', 'logg', 'rad']
        
        for idx, row in enumerate(data):
            missing_fields = [field for field in required_fields if field not in row]
            if missing_fields:
                return jsonify({
                    "error": f"Row {idx} missing required fields: {missing_fields}"
                }), 400
        
        # Process each row with the model
        results = []
        for row in data:
            try:

# result_type = modelResult["prediction"].lower().replace(' ', '_')
#     result = {
#         "type": result_type,
#         "confidence": modelResult['probabilities'][modelResult['prediction']] * 100,
#         "title": title_map[result_type],
#         "description": description_map[result_type]
    # }

                prediction= service.predict(row, return_proba=True)
                
                # Add original data plus predictions
                result_row = row.copy()
                result_row['model_prediction'] = prediction['prediction'].lower().replace(' ', '_')
                result_row['confidence_score'] = prediction['probabilities'][prediction['prediction']] * 100
                results.append(result_row)
                
            except Exception as e:
                return jsonify({
                    "error": f"Error processing row: {str(e)}"
                }), 500
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Convert to bytes for sending
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode())
        csv_bytes.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.csv"
        
        # Send the CSV file
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Flask API is running!"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
