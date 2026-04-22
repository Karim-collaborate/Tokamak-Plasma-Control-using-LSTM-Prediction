from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import os

from LSTM_model import disruption_detection_model
from data import generate_synthetic_data 

X_train, y_train = generate_synthetic_data()

#normalize data
samples, timesteps, features = X_train.shape
X_reshaped = X_train.reshape(-1, features)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_train = X_scaled.reshape(samples, timesteps, features)


model_path = "lstm_model.keras"
    
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = disruption_detection_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

#save
model.save(model_path)

#visualize model accuracy
plt.plot(history.history['accuracy'])
plt.title("Model accuracy")
plt.show()
