
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load the saved model, scaler, and label encoder ---
@st.cache_resource
def load_artifacts():
    with open('best_classification_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Supply Chain Risk Classifier", layout="centered")
st.title("🚚 Supply Chain Risk Classification")
st.write("Enter the details below to predict the risk classification for a supply chain operation.")

# --- Input Fields ---
st.header("Feature Inputs")

# Define the input fields dynamically or manually for better control
# Based on the X features from the notebook

# Grouping inputs for better UI organization (optional, but good practice)
col1, col2, col3 = st.columns(3)

with col1:
    vehicle_gps_latitude = st.number_input("Vehicle GPS Latitude", value=40.375568, format="%.6f")
    fuel_consumption_rate = st.number_input("Fuel Consumption Rate", value=5.136512, format="%.6f")
    warehouse_inventory_level = st.number_input("Warehouse Inventory Level", value=985.716862, format="%.6f")
    handling_equipment_availability = st.number_input("Handling Equipment Availability", value=0.481294, format="%.6f")
    weather_condition_severity = st.number_input("Weather Condition Severity", value=0.500000, format="%.6f")
    supplier_reliability_score = st.number_input("Supplier Reliability Score", value=0.500000, format="%.6f")
    iot_temperature = st.number_input("IoT Temperature", value=0.574400, format="%.6f")
    route_risk_level = st.number_input("Route Risk Level", value=1.182116, format="%.6f")
    driver_behavior_score = st.number_input("Driver Behavior Score", value=0.033843, format="%.6f")

with col2:
    vehicle_gps_longitude = st.number_input("Vehicle GPS Longitude", value=-77.014318, format="%.6f")
    eta_variation_hours = st.number_input("ETA Variation (Hours)", value=4.998009, format="%.6f")
    loading_unloading_time = st.number_input("Loading/Unloading Time", value=4.951392, format="%.6f")
    order_fulfillment_status = st.number_input("Order Fulfillment Status", value=0.761166, format="%.6f")
    port_congestion_level = st.number_input("Port Congestion Level", value=0.500000, format="%.6f")
    lead_time_days = st.number_input("Lead Time (Days)", value=1.000000, format="%.6f")
    cargo_condition_status = st.number_input("Cargo Condition Status", value=0.777263, format="%.6f")
    customs_clearance_time = st.number_input("Customs Clearance Time", value=0.502006, format="%.6f")
    fatigue_monitoring_score = st.number_input("Fatigue Monitoring Score", value=0.978599, format="%.6f")

with col3:
    traffic_congestion_level = st.number_input("Traffic Congestion Level", value=5.927586, format="%.6f")
    shipping_costs = st.number_input("Shipping Costs", value=456.503853, format="%.6f")
    historical_demand = st.number_input("Historical Demand", value=100.772854, format="%.6f")
    disruption_likelihood_score = st.number_input("Disruption Likelihood Score", value=0.506152, format="%.6f")
    delay_probability = st.number_input("Delay Probability", value=0.885291, format="%.6f")

# --- Prediction Button ---
if st.button("Predict Risk Classification"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([{
        'vehicle_gps_latitude': vehicle_gps_latitude,
        'vehicle_gps_longitude': vehicle_gps_longitude,
        'fuel_consumption_rate': fuel_consumption_rate,
        'eta_variation_hours': eta_variation_hours,
        'traffic_congestion_level': traffic_congestion_level,
        'warehouse_inventory_level': warehouse_inventory_level,
        'loading_unloading_time': loading_unloading_time,
        'handling_equipment_availability': handling_equipment_availability,
        'order_fulfillment_status': order_fulfillment_status,
        'weather_condition_severity': weather_condition_severity,
        'port_congestion_level': port_congestion_level,
        'shipping_costs': shipping_costs,
        'supplier_reliability_score': supplier_reliability_score,
        'lead_time_days': lead_time_days,
        'historical_demand': historical_demand,
        'iot_temperature': iot_temperature,
        'cargo_condition_status': cargo_condition_status,
        'route_risk_level': route_risk_level,
        'customs_clearance_time': customs_clearance_time,
        'driver_behavior_score': driver_behavior_score,
        'fatigue_monitoring_score': fatigue_monitoring_score,
        'disruption_likelihood_score': disruption_likelihood_score,
        'delay_probability': delay_probability
    }])

    # Ensure the order of columns matches the training data
    # (This is implicitly handled if you use the same feature names and order as X_train)
    # If the original X was a pandas DataFrame, we can get its columns directly.
    # For robustness, you might want to explicitly define the feature columns order.
    # For this example, we assume the order of input_data matches the model's expectation.

    # Scale the input features
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction_encoded = model.predict(scaled_input)

    # Inverse transform the prediction to get the original label
    predicted_risk_class = label_encoder.inverse_transform(prediction_encoded)

    st.subheader("Prediction Result")
    st.success(f"The predicted Risk Classification is: **{predicted_risk_class[0]}**")

st.markdown("---")
st.info("This application uses a Random Forest Classifier trained on your supply chain logistics dataset.")
