# Tokamak Plasma Control using LSTM Prediction

## 📌Overview

This project simulates an intelligent control system for a tokamak fusion reactor. It uses a Long Short-Term Memory (LSTM) neural network to predict plasma instabilities and potential disruptions, and adjusts control parameters in real time to maintain stable confinement.

The goal is to explore how machine learning can assist in managing complex plasma behavior in nuclear fusion systems.

---

## 🎯Problem Statement

Plasma in tokamak reactors is highly unstable and prone to disruptions, which can damage the reactor and reduce efficiency and lead to serious security problems.

This project addresses the problem by:

* Predicting instability patterns from time-series plasma data
* Reacting proactively instead of reactively
* Simulating automated control adjustments

---

## 🚀Features

* Time-series prediction of plasma behavior using LSTM
* Detection of potential instabilities and disruptions
* Automated adjustment of control parameters (plasma current)
* Simulation of a feedback control loop
* Visualization of predictions and system response

---

## 🛠️Tech Stack

* **Python**
* **TensorFlow / Keras** (LSTM model)
* **NumPy & scikit-learn** (data processing)
* **Matplotlib** (visualization)
* **simplepid** (control)
* **Scipy** (integrals)

---

## ⚙️How It Works

1. Plasma data (simulated) is fed into the LSTM model
2. The model predicts future plasma states
3. If instability is detected:

   * Control parameters are adjusted (e.g., current modification)
4. The system simulates the impact of these adjustments

---

## 📂Project Structure

```
├── src/
|   ├── physics.py      
|   ├── simulation.py
|   ├── control.py
|   └── models/
|       ├── data.py
|       ├── LSTM_model.py
│       └── training.py   
└── main.py              # Entry point
```

---

## ⚙️Installation

```bash
git clone https://github.com/Karim-collaborate/Tokamak-Plasma-Control-using-LSTM-Prediction.git
cd Tokamak-Plasma-Control-using-LSTM-Prediction
pip install tensorflow scipy scikit-learn numpy matplotlib simplepid
```

---

## ▶️Usage

```bash
python main.py
```

---

## 📊Results

The system demonstrates the ability to:

* Anticipate instability patterns
* Reduce simulated disruption events
* Improve overall plasma stability

---

## 💡Future Improvements

* Integration with real experimental datasets (e.g., ITER, JET)
* More advanced control strategies
* Real-time system deployment simulation
* Multi-parameter control (magnetic fields, pressure, etc.)

---

## ⚠️Disclaimer

This project is a simplified simulation for educational purposes and does not represent a real industrial control system.

---

## 👤Author
- Oolahiane karim
