from src.data import generate_synthetic_data 
from models.model import .  
import matplotlib.pyplot as plt 
# Model training 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train, y_train = generate_synthetic_data()
model = disruption_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save("models/lstm_model.keras")
plt.plot(history.history['loss'])

