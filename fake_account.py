from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the pre-trained model and scaler
model = load_model('model/fake_account_detection_model.h5')
scaler = joblib.load('model/scaler.pkl')  # Load the pre-trained scaler
label_encoders = {
    'lang': joblib.load('model/lang.pkl'),  # Assuming label encoder for lang is saved
    'sex': joblib.load('model/sex.pkl')  # Assuming label encoder for sex is saved
}

# Define feature columns
selected_features = [
    'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 
    'listed_count', 'lang', 'sex', 'last_update'
]

# Function to preprocess input data
def preprocess_input(data):
    data = pd.DataFrame([data], columns=selected_features)

    # Handle missing values by filling with the mean
    data = data.fillna(data.mean())

    # Encode categorical features (lang, sex) with validation
    if data['lang'][0] not in label_encoders['lang'].classes_:
        raise ValueError(f"Invalid value for lang: {data['lang'][0]}")
    if data['sex'][0] not in label_encoders['sex'].classes_:
        raise ValueError(f"Invalid value for sex: {data['sex'][0]}")

    data['lang'] = label_encoders['lang'].transform([data['lang'][0]])[0]
    data['sex'] = label_encoders['sex'].transform([data['sex'][0]])[0]

    # Normalize features
    scaled_data = scaler.transform(data)
    return scaled_data

# Function to predict fake or real account
def predict_account(data):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    return "Fake Account" if prediction[0][0] > 0.5 else "Real Account"

# Function to handle prediction
def handle_predict():
    try:
        # Get user input
        statuses_count = int(statuses_count_entry.get())
        followers_count = int(followers_count_entry.get())
        friends_count = int(friends_count_entry.get())
        favourites_count = int(favourites_count_entry.get())
        listed_count = int(listed_count_entry.get())
        lang = lang_entry.get().strip()
        sex = sex_entry.get().strip()
        last_update = int(last_update_entry.get())

        # Validate inputs for categorical fields
        if lang not in label_encoders['lang'].classes_:
            raise ValueError(f"Invalid value for lang: {lang}. Valid options: {label_encoders['lang'].classes_}")
        if sex not in label_encoders['sex'].classes_:
            raise ValueError(f"Invalid value for sex: {sex}. Valid options: {label_encoders['sex'].classes_}")

        # Prepare data for prediction
        input_data = [statuses_count, followers_count, friends_count, favourites_count, listed_count, lang, sex, last_update]
        result = predict_account(input_data)

        # Display result
        messagebox.showinfo("Prediction Result", f"The account is classified as: {result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the Tkinter UI
root = tk.Tk()
root.title("Fake Account Detection")

# Labels and Entry widgets
tk.Label(root, text="Statuses Count:").grid(row=0, column=0, padx=10, pady=5)
statuses_count_entry = tk.Entry(root)
statuses_count_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Followers Count:").grid(row=1, column=0, padx=10, pady=5)
followers_count_entry = tk.Entry(root)
followers_count_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Friends Count:").grid(row=2, column=0, padx=10, pady=5)
friends_count_entry = tk.Entry(root)
friends_count_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Favourites Count:").grid(row=3, column=0, padx=10, pady=5)
favourites_count_entry = tk.Entry(root)
favourites_count_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Listed Count:").grid(row=4, column=0, padx=10, pady=5)
listed_count_entry = tk.Entry(root)
listed_count_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Language (lang):").grid(row=5, column=0, padx=10, pady=5)
lang_entry = tk.Entry(root)
lang_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Sex (sex):").grid(row=6, column=0, padx=10, pady=5)
sex_entry = tk.Entry(root)
sex_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Last Update (numeric):").grid(row=7, column=0, padx=10, pady=5)
last_update_entry = tk.Entry(root)
last_update_entry.grid(row=7, column=1, padx=10, pady=5)

# Predict Button
tk.Button(root, text="Predict", command=handle_predict).grid(row=8, column=0, columnspan=2, pady=20)

# Run the application
root.mainloop()
