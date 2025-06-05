import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Check for required packages
try:
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow is not installed. Please install it using: pip install tensorflow")
    st.stop()

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
except ImportError:
    st.error("Scikit-learn is not installed. Please install it using: pip install scikit-learn")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-likely {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .churn-unlikely {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load files with error handling
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and preprocessing objects"""
    try:
        # Check if files exist
        required_files = ['model.h5', 'label_encoder_gender.pkl', 'onehot_encoder_geo.pkl', 'scaler.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            st.info("Please ensure all model files are in the same directory as this script.")
            st.stop()
        
        # Load the trained model
        model = tf.keras.models.load_model('model.h5')
        
        # Load the encoders and scaler
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load model and preprocessing objects
model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# Main title
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Information")
    
    # Geography and Gender
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    
    # Numeric inputs
    age = st.slider('🎂 Age', min_value=18, max_value=92, value=35)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('💰 Account Balance ($)', min_value=0.0, value=50000.0, format="%.2f")

with col2:
    st.subheader("📊 Account Details")
    
    estimated_salary = st.number_input('💵 Estimated Salary ($)', min_value=0.0, value=50000.0, format="%.2f")
    tenure = st.slider('⏱️ Tenure (years)', min_value=0, max_value=10, value=5)
    num_of_products = st.slider('🛍️ Number of Products', min_value=1, max_value=4, value=2)
    
    # Boolean inputs with better labels
    has_cr_card = st.selectbox('💳 Has Credit Card', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('🔥 Is Active Member', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Add some spacing
st.markdown("---")

# Prediction button
if st.button('🔮 Predict Churn Probability', type="primary"):
    try:
        # Prepare the input data
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
        
        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display results with better formatting
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Probability display
            st.metric(
                label="Churn Probability",
                value=f"{prediction_proba:.1%}",
                delta=f"{prediction_proba:.3f} raw score"
            )
            
            # Prediction result with styling
            if prediction_proba > 0.5:
                st.markdown(
                    f'<div class="prediction-box churn-likely">'
                    f'<h3>⚠️ High Risk Customer</h3>'
                    f'<p>This customer is <strong>likely to churn</strong> with a probability of <strong>{prediction_proba:.1%}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box churn-unlikely">'
                    f'<h3>✅ Low Risk Customer</h3>'
                    f'<p>This customer is <strong>unlikely to churn</strong> with a probability of <strong>{prediction_proba:.1%}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        # Additional insights
        st.subheader("📈 Model Insights")
        
        # Risk level categorization
        if prediction_proba >= 0.8:
            risk_level = "🔴 Very High Risk"
            recommendation = "Immediate intervention required. Consider offering incentives or personalized retention offers."
        elif prediction_proba >= 0.6:
            risk_level = "🟠 High Risk"
            recommendation = "Monitor closely and consider proactive engagement strategies."
        elif prediction_proba >= 0.4:
            risk_level = "🟡 Medium Risk"
            recommendation = "Regular check-ins and satisfaction surveys recommended."
        else:
            risk_level = "🟢 Low Risk"
            recommendation = "Customer appears satisfied. Continue current service level."
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Risk Level:** {risk_level}")
        with col2:
            st.info(f"**Recommendation:** {recommendation}")
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check that all input values are valid and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 Powered by TensorFlow & Streamlit | Customer Churn Prediction Model</p>
        <p><em>Thank you for using our prediction service!</em></p>
        <p style='font-size: 0.8em; margin-top: 1rem; font-style: italic;'>
            Originally built by me, but enhanced with Claude's help to make it more beautiful! 
            Sometimes it's good to let AI add that extra polish ✨
        </p>
    </div>
    """,
    unsafe_allow_html=True
)