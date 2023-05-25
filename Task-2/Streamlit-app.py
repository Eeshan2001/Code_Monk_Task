from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import streamlit as st
import requests
import io
from PIL import Image

# Load the model
model = load_model('path_to_your_model.h5')

# Set the API endpoint URL
API_URL = 'http://localhost:5000/predict'

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()

    # Preprocess the image data (replace with your own image preprocessing logic)
    processed_image = preprocess_image(image_data)

    # Reshape the image to match the input shape of the model
    processed_image = processed_image.reshape(1, 224, 224, 3)

    # Perform the prediction
    predictions = model.predict(processed_image)

    # Get the predicted labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Return the predicted label as a JSON response
    response = {'predicted_label': str(predicted_labels[0])}
    return jsonify(response)

def preprocess_image(image_data):
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Resize the image to the desired input shape of the model
    image = image.resize((224, 224))
    
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    
    # Preprocess the image array
    preprocessed_image = preprocess_input(image_array)
    
    return preprocessed_image

def main():
    st.title("Fashion Item Classification")
    st.text("Upload an image and click the 'Predict' button")

    # Upload and preprocess the image
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform the prediction
        if st.button('Predict'):
            # Prepare the image data
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_data.seek(0)

            # Send a POST request to the API endpoint
            response = requests.post(API_URL, files={'image': image_data})

            # Get the predicted label from the response
            predicted_label = response.json()['predicted_label']

            # Display the predicted label
            st.write('Predicted Label:', predicted_label)

if __name__ == '__main__':
    main()
