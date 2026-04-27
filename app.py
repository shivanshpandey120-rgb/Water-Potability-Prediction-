import streamlit as st

try:
    import pandas as pd
    import pickle
    import numpy as np
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.stop()

st.set_page_config(page_title="Water Potability Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'water_potability.csv')

# Function to train and save model if it doesn't exist
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(SCALER_PATH, 'rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            st.success("✓ Model loaded from cache.")
            return model, scaler
        except Exception as e:
            st.warning(f"Failed to load cached model: {e}. Training new model...")
    
    # Train model if files don't exist or failed to load
    st.info("🔄 Training model... This may take a moment.")
    
    try:
        # Load and prepare data
        water_data = pd.read_csv(DATA_PATH)
        X = water_data.drop('Potability', axis=1)
        y = water_data['Potability']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Try to save model and scaler (may fail on read-only systems)
        try:
            with open(MODEL_PATH, 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(SCALER_PATH, 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)
            st.success("✓ Model trained and saved successfully!")
        except Exception as save_error:
            st.warning(f"Could not save model files: {save_error}. Model is still available in memory.")
        
        return model, scaler
    
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.info("Make sure 'water_potability.csv' is in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to train model: {e}")
        st.stop()

# Load or train the model and scaler
try:
    model, scaler = load_or_train_model()
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.info("Make sure 'water_potability.csv' is in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()

st.title("Water Potability Prediction")
st.write("Enter the water quality metrics below to check if the water is potable.")

# Create input fields for the features used in the project
ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", value=200.0)
solids = st.number_input("Solids (ppm)", value=20000.0)
chloramines = st.number_input("Chloramines (ppm)", value=7.0)
sulfate = st.number_input("Sulfate (mg/L)", value=300.0)
conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# Prediction logic
if st.button("Predict Potability"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.success("The water is Potable (Safe to drink).")
    else:
        st.error("The water is Not Potable (Unsafe to drink).")
