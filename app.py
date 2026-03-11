import streamlit as st
import pandas as pd
import joblib
import os

# Load Model
model_path = "model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"File not found: {model_path}")
    st.stop()

st.title("Heart Attack Prediction App")

# Add all missing inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 50)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

with col2:
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

if st.button("Predict"):
    # 1. Load the "brain" files you just saved
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("preprocessor.joblib")

    # 2. Create the raw data from inputs
    raw_df = pd.DataFrame([{
        "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak,
        "cp": cp, "restecg": restecg, "slope": slope, "thal": thal, "ca": ca, 
        "sex": sex, "fbs": fbs, "exang": exang
    }])

    # 3. Apply the SAVED encoder to categorical columns
    cat_cols = ['cp', 'restecg', 'slope', 'thal', 'ca', 'sex', 'fbs', 'exang']
    encoded_cats = encoder.transform(raw_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))

    # 4. Apply the SAVED scaler to numerical columns
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    raw_df[num_cols] = scaler.transform(raw_df[num_cols])

    # 5. Combine them into the final input
    # Drop old cat cols and add the new encoded ones
    input_final = pd.concat([raw_df[num_cols], encoded_df], axis=1)

    # 6. Predict! 
    # (Since we used the real encoder, the column order will be 100% correct)
    prediction = model.predict(input_final)
    
    if prediction[0] == 1:
        st.error("Prediction: High Risk of Heart Attack")
    else:

        st.success("Prediction: Low Risk of Heart Attack")
