from tensorflow.keras.models import load_model
from src.data import generate_synthetic_data 
import matplotlib.pyplot as plt 
from src.models.LSTM_model import disruption_detection_model
import os

X_train, y_train = generate_synthetic_data()
model_path = "src/models/lstm_model.keras"
    
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
