import numpy as np

def generate_synthetic_data(samples=1000, timesteps=10):
    np.random.seed(42)

    X = []
    y = []

    for _ in range(samples):
        # Generate time series
        temperature = np.random.normal(10, 2, timesteps)   # keV
        density = np.random.normal(5, 1, timesteps)        # arbitrary units
        current = np.random.normal(1e4, 2e3, timesteps)    # Amperes
        noise = np.random.normal(0, 0.5, timesteps)

        # Stack features → shape (timesteps, features)
        sample = np.stack([temperature, density, current, noise], axis=1)

        # Define disruption rule (IMPORTANT PART)
        risk = (
            0.3 * np.mean(temperature) +
            0.3 * np.mean(density) +
            0.00005 * np.mean(current)
        )

        # Add randomness
        risk += np.random.normal(0, 0.5)

        # Convert to probability
        prob = 1 / (1 + np.exp(-risk))

        # Label
        label = 1 if prob > 0.5 else 0

        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)
