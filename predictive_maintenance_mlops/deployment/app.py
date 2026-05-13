
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; color: #e85d04; margin-bottom: 2rem; }
    .prediction-box { padding: 1rem; margin: 1rem 0; border-radius: 10px; text-align: center; }
    .failure-prediction { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .healthy-prediction { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="shashidj/Predictive-Maintenance-Model",
            filename="model.pkl"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_engine_condition(model, input_data):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    st.markdown('<div class="main-header">🔧 Predictive Maintenance — Engine Health Monitor</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check your configuration.")
        return

    st.sidebar.header("📊 Model Information")
    st.sidebar.info("""
    **Algorithm**: Best performing model from 6 algorithms  
    **Features**: Engine sensor readings  
    **Purpose**: Predict engine failure risk  
    **Accuracy**: Optimized through hyperparameter tuning
    """)

    st.header("Engine Sensor Readings")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔩 Mechanical Parameters")
        engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=5000.0, value=700.0, step=10.0)
        lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
        fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=50.0, value=11.8, step=0.1)

    with col2:
        st.subheader("🌡️ Temperature & Cooling")
        coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=3.2, step=0.1)
        lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=200.0, value=84.0, step=0.5)
        coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=81.6, step=0.5)

    if st.button("🔮 Predict Engine Condition", type="primary"):
        input_data = pd.DataFrame({
            'Engine rpm':         [engine_rpm],
            'Lub oil pressure':   [lub_oil_pressure],
            'Fuel pressure':      [fuel_pressure],
            'Coolant pressure':   [coolant_pressure],
            'lub oil temp':       [lub_oil_temp],
            'Coolant temp':       [coolant_temp]
        })

        prediction, prediction_proba = predict_engine_condition(model, input_data)

        if prediction is not None:
            st.header("🎯 Prediction Results")
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box failure-prediction">
                    <h3>⚠️ ENGINE FAILURE RISK DETECTED</h3>
                    <p><strong>Failure Probability:</strong> {prediction_proba[1]:.2%}</p>
                    <p>Immediate maintenance inspection is recommended!</p>
                </div>
                """, unsafe_allow_html=True)
                st.error("💡 Action Required: Schedule maintenance before next operation cycle.")
            else:
                st.markdown(f"""
                <div class="prediction-box healthy-prediction">
                    <h3>✅ ENGINE IS HEALTHY</h3>
                    <p><strong>Healthy Probability:</strong> {prediction_proba[0]:.2%}</p>
                    <p>Engine is operating within normal parameters.</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("💡 No immediate action required. Continue scheduled monitoring.")

            fig = go.Figure(data=[go.Bar(
                x=['Healthy', 'Failure Risk'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['#90ee90', '#ff7f7f']
            )])
            fig.update_layout(
                title="Engine Condition Probability Distribution",
                xaxis_title="Condition",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Sensor Reading Summary")
            summary_df = pd.DataFrame({
                'Sensor': ['Engine RPM', 'Lub Oil Pressure', 'Fuel Pressure',
                           'Coolant Pressure', 'Lub Oil Temp', 'Coolant Temp'],
                'Value': [f"{engine_rpm:.1f} RPM", f"{lub_oil_pressure:.2f} bar",
                          f"{fuel_pressure:.2f} bar", f"{coolant_pressure:.2f} bar",
                          f"{lub_oil_temp:.1f} °C", f"{coolant_temp:.1f} °C"]
            })
            st.table(summary_df)

    st.markdown("---")
    st.markdown("""
    **About this Application:**  
    This Predictive Maintenance system uses machine learning to monitor engine sensor data and predict failure risk in real time.  
    Built with MLOps best practices including experiment tracking, model versioning, and automated deployment.

    **🔧 Technical Stack:** Python • Scikit-learn • MLflow • HuggingFace • Streamlit • Docker
    """)

if __name__ == "__main__":
    main()
