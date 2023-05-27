from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
import requests

app = Flask(__name__)

# Path to the saved model file
model_path = "path_to_saved_model.h5"  # Replace with the actual model path

# Load the model
model = load_model(model_path)

# Endpoint for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Load and preprocess the image
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the predictions
    decoded_predictions = []
    for i in range(len(predictions)):
        decoded_predictions.append(np.argmax(predictions[i]))

    # Return the predicted labels as JSON response
    return jsonify({
        'baseColour': decoded_predictions[0],
        'articleType': decoded_predictions[1],
        'season': decoded_predictions[2],
        'gender': decoded_predictions[3]
    })

# Streamlit GUI application for model demonstration
st.title("Fashion Product Prediction")

# File uploader for uploading the image
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Button to trigger the prediction
if st.button("Predict") and file is not None:
    # Make a POST request to the API endpoint for prediction
    response = requests.post('http://localhost:5000/predict', files={'file': file})

    if response.status_code == 200:
        predictions = response.json()

        # Display the predicted labels
        st.subheader("Predicted Labels:")
        st.write("Base Colour:", predictions['baseColour'])
        st.write("Article Type:", predictions['articleType'])
        st.write("Season:", predictions['season'])
        st.write("Gender:", predictions['gender'])
    else:
        st.error("Error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
