from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"  # Update this if the model is in a different folder
with open(MODEL_PATH, "rb") as file:
    model, vectorizer = pickle.load(file)  # Adjust based on your model type

# Define a route to make predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.json
        symptoms = data.get("symptoms")  # Expecting symptoms as a string

        # Convert input into model format
        symptoms_vectorized = vectorizer.transform([symptoms])  # Assuming vectorizer was saved

        # Make a prediction
        prediction = model.predict(symptoms_vectorized)

        # Return the result as JSON
        return jsonify({"predicted_medicine": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)