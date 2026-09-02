import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import base64

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('LogisticRegression.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure all model files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.stop()

model, scaler, columns = load_models()

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.markdown("## ❤️ Heart Health Analyzer")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Prediction", "Risk Factors", "About"],
        icons=["activity", "clipboard-data", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#ff4b4b", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )

# Main header
st.markdown('<div class="main-header">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI-powered prediction for early detection of cardiovascular disease</div>', unsafe_allow_html=True)

if selected == "Prediction":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create tabs for different input categories
        tab1, tab2, tab3 = st.tabs(["📊 Demographics", "💉 Clinical Measurements", "⚡ ECG & Symptoms"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("📅 Age", 18, 100, 40, help="Age in years")
                sex = st.radio("⚥ Sex", ["M", "F"], horizontal=True)
                fasting_bs = st.radio("🍬 Fasting Blood Sugar > 120 mg/dL", [0, 1], 
                                    format_func=lambda x: "No" if x == 0 else "Yes",
                                    horizontal=True)
            with col_b:
                st.metric("Age Range", f"{age} years", delta=None)
                st.metric("Sex", sex)
                st.metric("Fasting BS", "Normal" if fasting_bs == 0 else "High")
        
        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                resting_bp = st.number_input("💓 Resting BP (mm Hg)", 80, 200, 120, 
                                           help="Resting blood pressure in mm Hg")
                cholesterol = st.number_input("🩸 Cholesterol (mg/dL)", 100, 600, 200,
                                            help="Serum cholesterol in mg/dL")
            with col_b:
                max_hr = st.slider("🏃 Max Heart Rate", 60, 220, 150, 
                                 help="Maximum heart rate achieved")
                oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1,
                                  help="ST depression induced by exercise relative to rest")
        
        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                chest_pain = st.selectbox("🫀 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                                        help="ATA: Atypical Angina, NAP: Non-anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
                resting_ecg = st.selectbox("📊 Resting ECG", ["Normal", "ST", "LVH"],
                                         help="ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy")
            with col_b:
                exercise_angina = st.radio("🏋️ Exercise-Induced Angina", ['Y', 'N'], horizontal=True)
                st_slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"],
                                      help="Slope of the peak exercise ST segment")
        
        # Create prediction button
        if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
            # Prepare input data
            raw_input = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_' + sex: 1,
                'ChestPainType_' + chest_pain: 1,
                'RestingECG_' + resting_ecg: 1,
                'ExerciseAngina_' + exercise_angina: 1,
                'ST_Slope_' + st_slope: 1
            }
            
            # Create DataFrame with all columns
            input_df = pd.DataFrame([raw_input])
            
            # Ensure all columns exist
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns
            input_df = input_df[columns]
            
            try:
                # Scale and predict
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]
                prediction_proba = model.predict_proba(scaled_input)[0]
                
                # Display results with animations
                st.markdown("---")
                
                if prediction == 1:
                    st.markdown(f'''
                    <div class="prediction-box high-risk">
                        ⚠️ High Risk of Heart Disease<br>
                        <span style="font-size: 1rem;">Confidence: {prediction_proba[1]:.1%}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Risk factors analysis
                    st.warning("🚨 **Risk Factors Identified:**")
                    risk_factors = []
                    if age > 55:
                        risk_factors.append(f"Age ({age} years) - increased risk")
                    if resting_bp > 140:
                        risk_factors.append(f"High Resting BP ({resting_bp} mm Hg)")
                    if cholesterol > 240:
                        risk_factors.append(f"High Cholesterol ({cholesterol} mg/dL)")
                    if max_hr < 100:
                        risk_factors.append(f"Low Max HR ({max_hr} bpm)")
                    if oldpeak > 2.0:
                        risk_factors.append(f"High Oldpeak ({oldpeak})")
                    if chest_pain == "ASY":
                        risk_factors.append("Asymptomatic chest pain")
                    if exercise_angina == 'Y':
                        risk_factors.append("Exercise-induced angina")
                    
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                    
                    st.info("💡 **Recommendation:** Please consult a cardiologist for a comprehensive evaluation.")
                    
                else:
                    st.markdown(f'''
                    <div class="prediction-box low-risk">
                        ✅ Low Risk of Heart Disease<br>
                        <span style="font-size: 1rem;">Confidence: {prediction_proba[0]:.1%}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.success("🎉 **Great News!** Your cardiovascular health indicators suggest low risk.")
                    st.info("💡 **Keep it up!** Maintain a healthy lifestyle with regular exercise and balanced diet.")
                
                # Display probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Probability (%)"},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "darkred" if prediction_proba[1] > 0.5 else "darkgreen"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'lightgreen'},
                            {'range': [30, 60], 'color': 'yellow'},
                            {'range': [60, 100], 'color': 'red'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Prediction error: {str(e)}")
                st.info("Please check all inputs and try again.")

elif selected == "Risk Factors":
    st.header("📋 Understanding Heart Disease Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Modifiable Risk Factors")
        st.markdown("""
        - **High Blood Pressure**: Maintain below 120/80 mm Hg
        - **High Cholesterol**: Keep LDL below 100 mg/dL
        - **Smoking**: Quit smoking immediately
        - **Physical Inactivity**: Exercise 30 min/day, 5 days/week
        - **Obesity**: Maintain BMI between 18.5-24.9
        - **Diabetes**: Control blood sugar levels
        """)
        
        st.subheader("🟡 Lifestyle Recommendations")
        st.markdown("""
        - 🥗 Eat heart-healthy diet (Mediterranean diet)
        - 🏃 Regular aerobic exercise
        - 🧘 Stress management techniques
        - 😴 Adequate sleep (7-8 hours)
        - 🍷 Limit alcohol consumption
        """)
    
    with col2:
        st.subheader("🔴 Non-Modifiable Risk Factors")
        st.markdown("""
        - **Age**: Risk increases with age
        - **Gender**: Men have higher risk
        - **Family History**: Genetic predisposition
        - **Race/Ethnicity**: Certain groups have higher risk
        """)
        
        st.subheader("📊 Key Health Metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Optimal BP", "120/80", delta="mm Hg")
            st.metric("Optimal Cholesterol", "<200", delta="mg/dL")
        with col_b:
            st.metric("Optimal Fasting BS", "<100", delta="mg/dL")
            st.metric("Optimal Heart Rate", "60-100", delta="bpm")

else:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🧠 AI-Powered Heart Disease Prediction
    
    This application uses a **Logistic Regression** machine learning model trained on cardiovascular data to predict the risk of heart disease.
    
    #### 🔬 How It Works:
    1. **Input** your clinical and demographic data
    2. The model analyzes patterns in your data
    3. **Predicts** the probability of heart disease
    4. Provides **personalized recommendations**
    
    #### 📊 Model Performance:
    - **Accuracy**: High accuracy on validation data
    - **Features**: 11 clinical features analyzed
    - **Training Data**: Comprehensive cardiovascular dataset
    
    #### ⚠️ Important Notice:
    This tool is for **educational and informational purposes only**. It should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.
    
    #### 🛠️ Technical Details:
    - Built with **Streamlit**
    - Uses **Scikit-learn** for ML
    - Real-time predictions
    - Interactive visualizations
    """)
    
    st.markdown("---")
    st.markdown("**❤️ Take care of your heart today for a healthier tomorrow!**")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Made with ❤️ | For educational purposes only | Data Privacy Protected</p>
</div>
""", unsafe_allow_html=True)