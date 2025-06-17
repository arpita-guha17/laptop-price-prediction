import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained pipeline
with open("model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Laptop Price Predictor")

# Input features (based on your dataset)
company = st.selectbox("Company", [
    'Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
    'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
    'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'
])
type_name = st.selectbox("Type", ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'])
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
ips = st.selectbox("IPS Display", ['No', 'Yes'])
screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, step=0.1)
resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800'])

# Convert resolution to PPI
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res*2 + y_res*2) ** 0.5) / screen_size

cpu_brand = st.selectbox("CPU Brand", ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Other Intel Processor', 'AMD Processor'])
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
gpu_brand = st.selectbox("GPU Brand", ['Intel', 'Nvidia', 'AMD'])
os = st.selectbox("Operating System", ['Windows', 'Mac', 'Linux', 'Chrome OS', 'No OS', 'Others'])

# Convert binary categorical to numeric
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Prepare input for prediction
input_df = pd.DataFrame([[
    company, type_name, ram, weight, touchscreen, ips, ppi,
    cpu_brand, hdd, ssd, gpu_brand, os
]], columns=[
    'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
    'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'
])

# Predict button
if st.button("Predict Laptop Price"):
    prediction = model.predict(input_df)[0]
    st.success(f" Estimated Price: ₹ {round(prediction * 100000, 2)}")  # Assuming model was trained in lakhs