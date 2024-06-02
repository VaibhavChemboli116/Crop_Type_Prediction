import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("soil_measures.csv")

X = df.drop("crop", axis=1).values 
y = df["crop"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  
random_state=116)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

loaded_xgb_model = joblib.load("best_xgb_model.joblib")

def predict_crop(N, P, K, pH):
    test_data = np.array([N, P, K, pH]).reshape(1, -1)
    prediction = loaded_xgb_model.predict(test_data)
    original_predictions = label_encoder.inverse_transform(prediction)
    return original_predictions[0]

def main():
    st.title('Crop Type Prediction🌾')

    st.header('Input Features')
    N = st.number_input('Nitrogen (N)', value=100.0)
    P = st.number_input('Phosphorus (P)', value=50.0)
    K = st.number_input('Potassium (K)', value=150.0)
    pH = st.number_input('pH', value=7.0)

    if st.button('Predict'):
        st.markdown('<hr>', unsafe_allow_html=True)
        crop_type = predict_crop(N, P, K, pH)
        st.header('Predicted Crop Type ')
        st.write(crop_type)
if __name__ == '__main__':
    main()
