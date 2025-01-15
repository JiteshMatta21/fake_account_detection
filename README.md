# fake_account_detection
This project implements a machine learning model to detect fake social media accounts using a variety of user activity metrics. The application includes data preprocessing, model training, and a user-friendly UI for predictions using Tkinter.

Features

Preprocessing: Handles missing values, encodes categorical features, and scales numerical data for optimal model performance.
Model Training: Utilizes a fully connected neural network built with TensorFlow/Keras.
Saved Artifacts: Saves the trained model, scalers, and label encoders for reuse.
Prediction UI: Provides a Tkinter-based interface for real-time predictions.

Model Training

Steps

Prepare the dataset (dataset.csv) with the following columns:
statuses_count
followers_count
friends_count
favourites_count
listed_count
lang (categorical)
sex (categorical)
last_update
Fake_Profile (binary target variable)

Run the model training script:
python main.py

The following artifacts will be saved in the model/ directory:
fake_account_detection_model.h5: The trained model.
scaler.pkl: Scaler for feature normalization.
lang_encoder.pkl: Label encoder for the lang feature.
sex_encoder.pkl: Label encoder for the sex feature.

