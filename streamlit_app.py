import streamlit as st
import pandas as pd
import joblib
from feature_engineering import FeatureEngineer

# Load model and required features
model = joblib.load("xgb_house_price_model.joblib")
required_cols = joblib.load("feature_columns.joblib")

st.title("🏠 House Price Prediction App")
st.markdown("### Fill in the house details below:")

# User Inputs
user_input = {
    "PoolQC": st.selectbox("Pool Quality", ["None", "Ex", "Gd", "Fa"]),
    "Fence": st.selectbox("Fence Quality", ["None", "GdWo", "MnPrv", "GdPrv", "MnWw"]),
    "FireplaceQu": st.selectbox("Fireplace Quality", ["None", "Ex", "Gd", "TA", "Fa", "Po"]),
    "YrSold": st.number_input("Year Sold", 2006, 2024, 2010),
    "YearBuilt": st.number_input("Year Built", 1900, 2023, 2000),
    "YearRemodAdd": st.number_input("Remodel Year", 1900, 2023, 2005),
    "GarageYrBlt": st.number_input("Garage Built Year", 1900, 2023, 2000),
    "MoSold": st.slider("Month Sold", 1, 12, 6),
    "OpenPorchSF": st.number_input("Open Porch SF", 0, 500, 50),
    "EnclosedPorch": st.number_input("Enclosed Porch SF", 0, 500, 0),
    "3SsnPorch": st.number_input("3 Season Porch SF", 0, 500, 0),
    "ScreenPorch": st.number_input("Screen Porch SF", 0, 500, 0),
    "BsmtFullBath": st.slider("Basement Full Baths", 0, 3, 1),
    "BsmtHalfBath": st.slider("Basement Half Baths", 0, 2, 0),
    "FullBath": st.slider("Full Baths", 0, 3, 2),
    "HalfBath": st.slider("Half Baths", 0, 2, 1),
    "TotalBsmtSF": st.number_input("Total Basement SF", 0, 3000, 800),
    "1stFlrSF": st.number_input("1st Floor SF", 0, 3000, 1000),
    "2ndFlrSF": st.number_input("2nd Floor SF", 0, 3000, 500),
    "WoodDeckSF": st.number_input("Wood Deck SF", 0, 1000, 100),
    "OverallQual": st.slider("Overall Quality (1–10)", 1, 10, 5),
    "OverallCond": st.slider("Overall Condition (1–10)", 1, 10, 5),
    "CentralAir": st.selectbox("Central Air", ["Y", "N"]),
    "PavedDrive": st.selectbox("Paved Drive", ["Y", "P", "N"]),
    "Street": st.selectbox("Street Type", ["Pave", "Grvl"]),
    "Alley": st.selectbox("Alley Type", ["Pave", "Grvl", "NA"]),
    "MasVnrArea": st.number_input("Masonry Veneer Area", 0, 1000, 0),
    "GrLivArea": st.number_input("Above Ground Living Area (sq ft)", min_value=0, value=1500),
    "GarageCars": st.slider("Garage Cars", 0, 4, 2)
}

# Create DataFrame
input_df = pd.DataFrame([user_input])

# Fix types to avoid Categorical errors
for col in ['PoolQC', 'Fence', 'FireplaceQu', 'CentralAir', 'PavedDrive', 'Street', 'Alley']:
    input_df[col] = input_df[col].astype(str).fillna("None")

# Fill missing engineer-required columns if any (future-proofing)
engineer_required_cols = [
    'PoolQC', 'Fence', 'FireplaceQu', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
    'MoSold', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF',
    'OverallQual', 'OverallCond', 'CentralAir', 'PavedDrive', 'Street', 'Alley',
    'MasVnrArea'
]

for col in engineer_required_cols:
    if col not in input_df.columns:
        input_df[col] = "None" if col in ['PoolQC', 'Fence', 'FireplaceQu', 'CentralAir', 'PavedDrive', 'Street', 'Alley'] else 0

# Feature Engineering
fe = FeatureEngineer()
try:
    transformed_df = fe.transform(input_df)
except Exception as e:
    st.error(f"Feature Engineering failed: {e}")
    st.stop()

# Ensure all required features are present
for col in required_cols:
    if col not in transformed_df.columns:
        transformed_df[col] = 0

transformed_df = transformed_df[required_cols]

# Prediction
try:
    prediction = model.predict(transformed_df)[0]
    st.success(f"🏡 Predicted House Price: **${prediction:,.2f}**")
except Exception as e:
    st.error(f"Prediction failed: {e}")
