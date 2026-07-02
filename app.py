import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# This tells Python: find files relative to where app.py itself is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved model and columns
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'columns.pkl'), 'rb') as f:
    model_columns = pickle.load(f)

st.title("🏠 Rental Price Predictor")
st.write("Enter property details below to get an estimated monthly rent.")

# --- User Inputs ---
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"])
bhk = st.slider("BHK (Bedrooms)", 1, 6, 2)
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=800)
bathroom = st.slider("Bathrooms", 1, 6, 1)
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors/Family", "Bachelors", "Family"])
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
floor_number = st.number_input("Floor Number", min_value=0, max_value=50, value=1)
total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=3)

# --- Predict Button ---
if st.button("Predict Rent"):

    # Build input as a dataframe with all zeros
    input_dict = {col: 0 for col in model_columns}

    # Fill in numeric features directly
    # Fill in numeric features directly
    input_dict['Size'] = size
    input_dict['BHK'] = bhk
    input_dict['Bathroom'] = bathroom
    input_dict['Floor_Number'] = floor_number
    input_dict['Total_Floors'] = total_floors
    
    # Engineered features — must match exactly what notebook created
    input_dict['floor_ratio'] = floor_number / (total_floors + 1e-5)
    input_dict['bhk_per_bathroom'] = bhk / (bathroom + 1e-5)

    # Fill in one-hot encoded features
    # Only set to 1 if the column exists (drop_first means some are baseline)
    if f'City_{city}' in input_dict:
        input_dict[f'City_{city}'] = 1

    if f'Furnishing Status_{furnishing}' in input_dict:
        input_dict[f'Furnishing Status_{furnishing}'] = 1

    if f'Tenant Preferred_{tenant}' in input_dict:
        input_dict[f'Tenant Preferred_{tenant}'] = 1

    if f'Area Type_{area_type}' in input_dict:
        input_dict[f'Area Type_{area_type}'] = 1

    if f'Point of Contact_{contact}' in input_dict:
        input_dict[f'Point of Contact_{contact}'] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Predict (remember model predicts log rent)
    log_pred = model.predict(input_df)[0]

    # Convert back to actual ₹
    predicted_rent = np.expm1(log_pred)

    # Display result
    st.success(f"### Predicted Monthly Rent: ₹{predicted_rent:,.0f}")
    st.write(f"*(Range estimate: ₹{predicted_rent*0.85:,.0f} – ₹{predicted_rent*1.15:,.0f})*")
