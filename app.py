from flask import Flask, request, jsonify
from flask_cors import CORS  # allows React frontend to call Flask backend
from ModelClass import PredictionModel
import artifacts
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

service = PredictionModel(artifacts_dir="artifacts")
# // Expected API response format:
#       // {
#       //   type: 'confirmed' | 'candidate' | 'false_positive',
#       //   confidence: number (0-100),
#       //   title: string,
#       //   description: string
#       // }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    print("Received from frontend:", data) 
    # instantiate once at server startup
    
    modelResult = service.predict(data, return_proba=True)
    # result is a dict -> return as JSON to chatbot

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

    result={
            "type": modelResult["prediction"].lower().replace(' ','_'),
            "confidence": modelResult['probabilities'][modelResult['prediction']]*100,
            "title": title_map[modelResult["prediction"].lower().replace(' ','_')],
            "description": description_map[modelResult["prediction"].lower().replace(' ','_')]
    }  


    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 