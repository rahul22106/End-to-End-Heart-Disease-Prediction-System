import streamlit as st
import joblib 
import pandas as pd 
import numpy as np
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Predictor")
st.markdown("---")

# Check if model and scaler exist
model_path = 'artifacts/model_trainer/trained_model/heart_disease_model.joblib'
scaler_path = 'artifacts/data_transformation/standard_scaler.joblib'

if not os.path.exists(model_path):
    st.error("""
    ❌ **Model not found!**
    
    Please run the model training pipeline first to generate the heart disease prediction model.
    
    The model should be located at: `artifacts/model_trainer/trained_model/heart_disease_model.joblib`
    """)
    
    st.info("""
    **To train the model:**
    1. Run the data ingestion pipeline
    2. Run the data validation pipeline  
    3. Run the data transformation pipeline
    4. Run the model training pipeline
    """)
    
    # Create a placeholder for demonstration
    if st.button("Create Demo Model (for testing only)"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Create a simple demo model
        demo_model = LogisticRegression(random_state=42)
        
        # Create dummy training data matching your feature structure
        X_dummy = np.random.rand(100, 11)
        y_dummy = np.random.randint(0, 2, 100)
        
        demo_model.fit(X_dummy, y_dummy)
        
        # Create directory if it doesn't exist
        os.makedirs('artifacts/model_trainer/trained_model', exist_ok=True)
        joblib.dump(demo_model, model_path)
        
        # Create demo scaler
        demo_scaler = StandardScaler()
        demo_scaler.fit(X_dummy)
        os.makedirs('artifacts/data_transformation', exist_ok=True)
        joblib.dump(demo_scaler, scaler_path)
        
        st.success("✅ Demo model and scaler created! You can now test the interface.")
        st.rerun()
    
    st.stop()

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.stop()

def predict_heart_disease(input_data):
    """Make prediction using the trained model with scaling"""
    try:
        # Scale the input data using the fitted StandardScaler
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Create tabs
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", [
            "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"
        ])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl", "> 120 mg/dl"])
        resting_ecg = st.selectbox("Resting ECG Results", [
            "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"
        ])
        max_hr = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # Convert inputs to model format (matching your training pipeline)
    sex_encoded = 0 if sex == "Male" else 1
    
    # Chest pain encoding (matches your training data)
    chest_pain_mapping = {
        "Typical Angina": 3,
        "Atypical Angina": 0, 
        "Non-Anginal Pain": 1,
        "Asymptomatic": 2
    }
    chest_pain_encoded = chest_pain_mapping[chest_pain]
    
    fasting_bs_encoded = 1 if fasting_bs == "> 120 mg/dl" else 0
    
    # Resting ECG encoding
    resting_ecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    resting_ecg_encoded = resting_ecg_mapping[resting_ecg]
    
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    # ST Slope encoding
    st_slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    st_slope_encoded = st_slope_mapping[st_slope]
    
    # Create input dataframe (ensure column order matches training)
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_encoded],
        'ChestPainType': [chest_pain_encoded],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_encoded],
        'RestingECG': [resting_ecg_encoded],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_encoded],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_encoded]
    })
    
    if st.button("Predict Heart Disease", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            prediction = predict_heart_disease(input_data)
            
            if prediction is not None:
                st.markdown("---")
                st.header("Prediction Result")
                
                # Display only the final prediction
                if prediction == 1:
                    st.error("🚨 **Heart Disease Detected**")
                else:
                    st.success("✅ **No Heart Disease Detected**")

with tab2:
    st.header("Batch Prediction")
    
    st.info("Upload a CSV file with patient data for multiple predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_data.head())
            
            # Check if required columns exist
            required_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                              'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                              'Oldpeak', 'ST_Slope']
            
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"Missing columns in uploaded file: {missing_columns}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        # Scale the batch data
                        scaled_batch_data = scaler.transform(batch_data[required_columns])
                        predictions = model.predict(scaled_batch_data)
                        
                        # Add predictions to dataframe
                        results_df = batch_data.copy()
                        results_df['HeartDisease_Prediction'] = predictions
                        results_df['Result'] = np.where(
                            results_df['HeartDisease_Prediction'] == 1, 
                            'Heart Disease Detected', 
                            'No Heart Disease Detected'
                        )
                        
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Patients", len(results_df))
                        with col2:
                            heart_disease_count = (results_df['HeartDisease_Prediction'] == 1).sum()
                            st.metric("Heart Disease Cases", heart_disease_count)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="heart_disease_predictions.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Sidebar information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This Heart Disease Predictor uses a **Logistic Regression** model trained on clinical data to assess the risk of heart disease.
    
    **Model Features:**
    - Demographic information (Age, Sex)
    - Clinical measurements (BP, Cholesterol)
    - ECG results and exercise test data
    - Symptom information
    
    **Note:** This tool is for educational purposes and should not replace professional medical advice.
    """)
    
    st.markdown("---")
    st.header("Model Information")
    st.write(f"Model: **Logistic Reg*")
    st.write(f"Model Location: `{model_path}`")
    st.write(f"Scaler Location: `{scaler_path}`")
    
    if st.button("Check Model Status"):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            st.success("✅ Model and Scaler loaded successfully")
        else:
            st.error("❌ Model or Scaler file not found")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Heart Disease Prediction System | Logistic Regression Model | For Educational Purposes"
    "</div>", 
    unsafe_allow_html=True
)