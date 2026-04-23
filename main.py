import os
from tensorflow.keras.models import load_model
from src.models.LSTM_model import disruption_detection_model

from src.simulation import plasma_control_loop

model_path = "src/models/lstm_model.keras"

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("Results aren't accurate . Try Training model first")
    model = disruption_detection_model(input_shape=(10,4))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

plasma_control_loop(model)
