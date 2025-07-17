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
    st.header("📊 Log-Normal Distribution (Original + Normalized)")

    mu = st.number_input("Mean (mu):", value=0.9)
    sigma = st.number_input("Standard Deviation (sigma):", value=0.01)
    size = st.slider("Sample Size", 1000, 20000, 10000, step=1000)

    if st.button("Generate & Plot Distribution"):
        np.random.seed(42)
        e = np.e
        samples = np.random.lognormal(mean=mu, sigma=sigma, size=size)

        # Normalize with min=1 and max=e
        x_min, x_max = 1, e
        samples_clipped = np.clip(samples, x_min, x_max)
        samples_norm = (samples_clipped - x_min) / (x_max - x_min)

        # Calculate means
        mean_original = np.mean(samples)
        mean_normalized = np.mean(samples_norm)

        # Plot both
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].hist(samples, bins=50, color='cornflowerblue', edgecolor='black')
        axs[0].axvline(x=e, color='red', linestyle='dashed', linewidth=2, label='e ≈ 2.718, miu=1')
        axs[0].axvline(x=1, color='red', linestyle='dashed', linewidth=2, label='1 ≈ miu=0')
        axs[0].set_title(f'Original Log-Normal (μ={mu}, σ={sigma})')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].hist(samples_norm, bins=50, color='mediumseagreen', edgecolor='black')
        axs[1].set_title('Normalized Samples (min=1, max=e)')
        axs[1].set_xlabel('Normalized Value (0 to 1)')
        axs[1].set_ylabel('Frequency')
        axs[1].grid(True)

        st.pyplot(fig)

        # Show means
        st.subheader("📈 Mean Values")
        st.write(f"🔵 **Original Mean:** `{mean_original:.4f}`")
        st.write(f"🟢 **Normalized Mean (min=1, max=e → 0-1):** `{mean_normalized:.4f}`")

