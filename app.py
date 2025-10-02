from flask import Flask, request, jsonify
from flask_cors import CORS  # allows React frontend to call Flask backend
from ModelClass import PredictionModel
import artifacts

# 1. Initialize Flask app
app = Flask(__name__)

# 2. Configure CORS globally for all routes ("/*") and all origins ("*")
CORS(app, resources={r"/*": {"origins": "*"}})

# 3. Instantiate model once at server startup
# Note: Ensure that PredictionModel and artifacts are available in your PA file structure.
service = PredictionModel(artifacts_dir="artifacts")

# Expected API response format:
# {
#   type: 'confirmed' | 'candidate' | 'false_positive',
#   confidence: number (0-100),
#   title: string,
#   description: string
# }

# Simple test route (optional, but helpful for debugging)
@app.route("/", methods=["GET"])
def check_status():
    """A simple GET request to check if the server is alive and CORS is enabled."""
    return "Exoplanet Analysis API is running and CORS is configured."

# FIX: Handle the automatic browser request for favicon.ico
@app.route('/favicon.ico')
def favicon():
    """Returns a 204 No Content response to handle automatic favicon requests gracefully."""
    # This prevents the request from triggering a 500 error in the absence of a file.
    return '', 204

@app.route("/", methods=["POST"])
def analyze():
    """Handles the main POST request for exoplanet prediction."""
    try:
        data = request.get_json()
        print("Received from frontend:", data) 
        
        modelResult = service.predict(data, return_proba=True)

        title_map={
            "confirmed":"Confirmed Exoplanet",
            "candidate":"Exoplanet Candidate",
            "false_positive":"False Positive"
        }

        description_map={
            "confirmed":"Congratulations! The data strongly suggests a confirmed exoplanet detection. All parameters fall within expected ranges for a genuine planetary transit.",
            "candidate":"Promising signals detected. The data shows characteristics consistent with a planetary transit, but additional observations are recommended for confirmation.",
            "false_positive":"Analysis indicates this signal is likely caused by stellar activity, eclipsing binary stars, or instrumental effects rather than a genuine exoplanet."
        }

        # Normalize prediction keys to snake_case for consistent API response
        pred_key = modelResult["prediction"].lower().replace(' ','_')

        result={
            "type": pred_key,
            "confidence": modelResult['probabilities'][modelResult['prediction']]*100,
            "title": title_map[pred_key],
            "description": description_map[pred_key]
        }   

        return jsonify(result)

    except Exception as e:
        # Important: Log the error to the server's error log and return a 500 status.
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": "Internal Server Error during prediction or processing."}), 500

# 4. Remove app.run() for PythonAnywhere deployment!
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
