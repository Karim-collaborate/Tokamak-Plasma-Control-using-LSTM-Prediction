from src.data import generate_synthetic_data 
from LSTM_model import disruption_detection_model 
import matplotlib.pyplot as plt 


# Model training 
X_train, y_train = generate_synthetic_data()
model = disruption_detection_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save("models/lstm_model.keras")
plt.plot(history.history['loss'])

