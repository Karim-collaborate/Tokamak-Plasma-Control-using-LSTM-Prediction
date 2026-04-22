from tensorflow.keras.models import load_model
from src.data import generate_synthetic_data 
import matplotlib.pyplot as plt 


# Model training 
X_train, y_train = generate_synthetic_data()
model = load_model("src/models/lstm_model.keras")

history = model.fit(X_train, y_train, epochs=10, batch_size=32)

model.save("src/models/lstm_model.keras")

plt.plot(history.history['accuracy'])
plt.title("Model accuracy")
plt.show()
