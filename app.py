    if st.button("Generate & Plot Distributions"):
        np.random.seed(42)
        lognorm_data = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        neg_lognorm_data = -lognorm_data

        # Scale both
        scaler = MinMaxScaler()
        lognorm_scaled = scaler.fit_transform(lognorm_data.reshape(-1, 1)).flatten()
        neg_lognorm_scaled = scaler.fit_transform(neg_lognorm_data.reshape(-1, 1)).flatten()

        # Calculate means
        lognorm_mean = np.mean(lognorm_scaled)
        neg_lognorm_mean = np.mean(neg_lognorm_scaled)

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

        # Show mean values
        st.subheader("📈 Scaled Distribution Means")
        st.write(f"🔵 **Scaled Log-Normal Mean:** `{lognorm_mean:.4f}`")
        st.write(f"🔴 **Scaled Negative Log-Normal Mean:** `{neg_lognorm_mean:.4f}`")
