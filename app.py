import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Configure page for faster loading
st.set_page_config(page_title="Churn Prediction & Salary Estimation", layout="centered")

# Sidebar - About & Model Details
with st.sidebar:
    st.header("📋 About")
    st.write("**Customer Churn Prediction & Salary Estimation**")
    st.write("Predicts the likelihood of a customer leaving and estimates their salary based on their profile and banking behavior.")
    
    st.markdown("---")
    st.subheader("🤖 Churn Model Details")
    
    # Architecture
    st.write("**Architecture:** Sequential Neural Network")
    st.write("**Layers:** 3 Dense layers (64→32→1)")
    st.write("**Parameters:** 2,945 (11.5 KB)")
    st.write("**Activation:** ReLU + Sigmoid")
    
    # Training Details
    st.markdown("**Training Configuration:**")
    st.write("• Optimizer: Adam")
    st.write("• Loss: Binary Crossentropy") 
    st.write("• Metrics: Accuracy")
    st.write("• Total Epochs: 100")
    
    # Last Epoch Performance
    st.markdown("---")
    st.subheader("📈 Final Performance")
    st.write("**Epoch 15/100** (Best Performance)")
    
    # Metrics in clean format
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Acc", "86.5%")
        st.metric("Training Loss", "0.330")
    with col2:
        st.metric("Val Acc", "85.5%")  
        st.metric("Val Loss", "0.360")
    
    st.write("⚡ **Training Speed:** 6ms/step")
    st.write("🎯 **Model Status:** Production Ready")
    
    st.markdown("---")
    st.subheader("💰 Salary Model Details")
    st.write("**Type:** Regression Neural Network")
    st.write("**Target:** Estimated Salary")
    st.write("**Features:** Customer profile + geography")
    
    st.markdown("---")
    st.write("**Features Used:**")
    st.write("• Credit Score • Geography")
    st.write("• Gender • Age • Tenure")
    st.write("• Balance • Products")
    st.write("• Credit Card • Activity")

@st.cache_resource
def load_resources():
    """Load models and encoders once"""
    try:
        import tensorflow as tf
        
        # Load churn prediction model
        churn_model = tf.keras.models.load_model('model.h5')
        
        # Load salary regression model
        salary_model = tf.keras.models.load_model('regression_model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            onehot_encoder_geo = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return churn_model, salary_model, label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

# Load once
churn_model, salary_model, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()

# Simple, fast UI
st.title('🎯 Customer Churn Prediction & Salary Estimation')

# Add tabs for different predictions
tab1, tab2 = st.tabs(["🔮 Churn Prediction", "💰 Salary Estimation"])

with tab1:
    st.header("Customer Churn Prediction")
    
    # Quick form layout
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
            gender = st.selectbox('Gender', label_encoder_gender.classes_)
            age = st.number_input('Age', 18, 92, 35)
            balance = st.number_input('Balance', 0.0, format="%.2f")
            credit_score = st.number_input('Credit Score', 300, 850, 650)
        
        with col2:
            estimated_salary = st.number_input('Estimated Salary', 0.0, format="%.2f")
            tenure = st.number_input('Tenure (years)', 0, 10, 5)
            num_of_products = st.number_input('Number of Products', 1, 4, 1)
            has_cr_card = st.selectbox('Has Credit Card', [0, 1])
            is_active_member = st.selectbox('Is Active Member', [0, 1])
        
        # Single predict button
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            # Fast prediction
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            # One-hot encode geography
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine and scale
            final_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
            scaled_data = scaler.transform(final_data)
            
            # Predict
            prediction = churn_model.predict(scaled_data, verbose=0)[0][0]
            
            # Results
            if prediction > 0.5:
                st.error(f"⚠️ **HIGH CHURN RISK** ({prediction:.1%})")
            else:
                st.success(f"✅ **LOW CHURN RISK** ({prediction:.1%})")

with tab2:
    st.header("Salary Estimation")
    
    # Salary prediction form
    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sal_geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='sal_geo')
            sal_gender = st.selectbox('Gender', label_encoder_gender.classes_, key='sal_gender')
            sal_age = st.number_input('Age', 18, 92, 35, key='sal_age')
            sal_balance = st.number_input('Balance', 0.0, format="%.2f", key='sal_balance')
            sal_credit_score = st.number_input('Credit Score', 300, 850, 650, key='sal_credit')
        
        with col2:
            sal_tenure = st.number_input('Tenure (years)', 0, 10, 5, key='sal_tenure')
            sal_num_of_products = st.number_input('Number of Products', 1, 4, 1, key='sal_products')
            sal_has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='sal_card')
            sal_is_active_member = st.selectbox('Is Active Member', [0, 1], key='sal_active')
        
        # Predict salary button
        sal_submitted = st.form_submit_button("💰 Estimate Salary", use_container_width=True)
        
        if sal_submitted:
            # Prepare input data for salary prediction (same structure as churn model)
            sal_input_data = pd.DataFrame({
                'CreditScore': [sal_credit_score],
                'Gender': [label_encoder_gender.transform([sal_gender])[0]],
                'Age': [sal_age],
                'Tenure': [sal_tenure],
                'Balance': [sal_balance],
                'NumOfProducts': [sal_num_of_products],
                'HasCrCard': [sal_has_cr_card],
                'IsActiveMember': [sal_is_active_member],
                'EstimatedSalary': [0]  # Placeholder - we'll predict this
            })
            
            # One-hot encode geography
            sal_geo_encoded = onehot_encoder_geo.transform([[sal_geography]]).toarray()
            sal_geo_df = pd.DataFrame(sal_geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine data (same order as churn model)
            sal_final_data = pd.concat([sal_input_data.reset_index(drop=True), sal_geo_df], axis=1)
            
            # Scale the data
            sal_scaled_data = scaler.transform(sal_final_data)
            
            # Predict salary
            predicted_salary = salary_model.predict(sal_scaled_data, verbose=0)[0][0]
            
            # Display result
            st.success(f"💰 **Estimated Salary: ${predicted_salary:,.2f}**")

# Simple footer
st.markdown("---")
st.caption("Originally simple app, enhanced with Claude for better UX & deployment")