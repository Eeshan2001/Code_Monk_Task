## For this task I have created separate file for EDA and also built the API with Flask and Streamlit GUI.

# Request you to evaluate both the model. For Buiding the Model. I have attached two files. 
   1) First one is important one. I used pretrined Mobilenet v2 model. -> attached on ipynb 
   2) Second one is building sequential Model with CNN layer:
        ```
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        
        # Set the path to the dataset files
        dataset_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset'

        # Load the style.csv file
        df = pd.read_csv(dataset_path + '/styles.csv', on_bad_lines='skip')
       
        # Perform any necessary exploratory data analysis (EDA) on the dataset
        df['id'] = df['id'].astype(str)  # Convert 'id' column to string data type

        # Modify 'id' column to include the image file extension
        df['id'] = df['id'].apply(lambda x: x + '.jpg')
        df.head()
        
        df = df.dropna()
        # df['gender'] = df['gender'].astype('category')
        df.dtypes
        
        from sklearn.preprocessing import LabelEncoder
        # Encode categorical labels
        label_encoder = LabelEncoder()
        df['baseColour'] = label_encoder.fit_transform(df['baseColour'])
        df['articleType'] = label_encoder.fit_transform(df['articleType'])
        df['season'] = label_encoder.fit_transform(df['season'])
        df['gender'] = label_encoder.fit_transform(df['gender'])

        # Split the dataset into training and testing sets
        train_ratio = 0.8
        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        test_df = df[train_size:]
        print(len(train_df))
        print(len(test_df))
        
        # Preprocess the image data
        image_size = (224, 224)  # Adjust the size as needed

        # Define an image data generator
        datagen = ImageDataGenerator(rescale=1./255)

        # Prepare the training data
        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=dataset_path + '/images',
            x_col='id',
            y_col=['baseColour', 'articleType', 'season', 'gender'],
            target_size=image_size,
            batch_size=32,
            class_mode='raw')

        # Prepare the testing data
        test_generator = datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=dataset_path + '/images',
            x_col='id',
            y_col=['baseColour', 'articleType', 'season', 'gender'],
            target_size=image_size,
            batch_size=32,
            class_mode='raw')
            
        train_labels = np.concatenate([train_generator.next()[1] for _ in range(len(train_generator))])
        print("Train labels shape:", train_labels.shape)
        print("Train labels:", train_labels)

        valid_labels = np.concatenate([test_generator.next()[1] for _ in range(len(test_generator))])
        print("Valid labels shape:", valid_labels.shape)
        print("Valid labels:", valid_labels)

        print("Unique valid labels:", np.unique(test_generator.labels))
        print("Unique train labels:", np.unique(train_generator.labels))
        
        # Retrieve a batch of preprocessed images from the generator
        batch_images, _ = train_generator.next()  # check the training images

        batch_images2, _ = test_generator.next() 
        print("Batch images shape:", batch_images.shape)
        print("Batch images shape:", batch_images2.shape)
        
        # Build the deep learning model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='softmax'))  # 4 output categories: baseColour, articleType, season, gender
        
        # Compile the model
        losses = ['categorical_crossentropy'] * 4
        model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])
        # Train the model
        model.fit(train_generator, validation_data=test_generator, epochs=10)

        ```
## Encountered many errors and issues while building these model. One common error is with categorical label mismatch. Like Corrupted imagese, in valid files, etc
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
