import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.title("🔬 Normalization & Log-Normal Distribution Visualizer")

menu = st.sidebar.selectbox("Select Menu", ["Min-Max Normalization", "Log-Normal Distribution"])

if menu == "Min-Max Normalization":
    st.header("🔧 Min-Max Normalization Calculator")

    min_val = st.number_input("Enter min value:", value=0.0)
    max_val = st.number_input("Enter max value:", value=1.0)
    x_val = st.number_input("Enter x value to normalize:", value=0.5)

    if st.button("Calculate Normalized Value"):
        if max_val > min_val:
            norm = (x_val - min_val) / (max_val - min_val)
            st.success(f"✅ Normalized value: {norm:.4f}")
        else:
            st.error("⚠️ Max value must be greater than min value.")

elif menu == "Log-Normal Distribution":
    st.header("📊 Log-Normal vs Negative Log-Normal")

    mu = st.number_input("Mean (mu):", value=0.5)
    sigma = st.number_input("Standard Deviation (sigma):", value=0.1)
    size = st.slider("Sample Size", 1000, 20000, 10000, step=1000)

    if st.button("Generate & Plot Distributions"):
        np.random.seed(42)
        lognorm_data = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        neg_lognorm_data = -lognorm_data

        # Scale both
        scaler = MinMaxScaler()
        lognorm_scaled = scaler.fit_transform(lognorm_data.reshape(-1, 1)).flatten()
        neg_lognorm_scaled = scaler.fit_transform(neg_lognorm_data.reshape(-1, 1)).flatten()

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].hist(lognorm_scaled, bins=100, color='steelblue', edgecolor='black', alpha=0.7, density=True)
        ax[0].set_title("Scaled Log-Normal (0–1)")
        ax[0].set_xlabel("Value")
        ax[0].set_ylabel("Density")

        ax[1].hist(neg_lognorm_scaled, bins=100, color='salmon', edgecolor='black', alpha=0.7, density=True)
        ax[1].set_title("Scaled Negative Log-Normal (0–1)")
        ax[1].set_xlabel("Value")
        ax[1].set_ylabel("Density")

        st.pyplot(fig)
