from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# AI model to predict plasma disruptions
def disruption_detection_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Disruption probability
    ])
    return model


