To run the combined file that includes the Flask API and Streamlit GUI, follow these steps:

1) Make sure you have the necessary dependencies installed. You can install the required dependencies by running the following command in your terminal or command prompt:

    Copy code
    ```
    pip install flask tensorflow keras numpy streamlit requests
    ```
2) Save the combined code into a file with a .py extension, such as app.py.

3) Replace "path_to_saved_model.h5" with the actual path to your saved model file.

4) Open a terminal or command prompt and navigate to the directory where you saved the app.py file.

5) Run the Flask application by executing the following command:

    Copy code
    ```
    python app.py
    ```
    This will start the Flask server on http://localhost:5000.

6) Open a web browser and access http://localhost:5000. You should see the Streamlit GUI application.

   Use the file uploader in the Streamlit GUI to upload an image and click the "Predict" button. The predicted labels will be displayed below.
