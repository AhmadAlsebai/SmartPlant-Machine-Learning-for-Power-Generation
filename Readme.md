## Dataset Overview
The CCPP dataset contains data collected from a power plant, where the goal is to predict the net hourly electrical energy output (PE) based on four input variables:

AT (Ambient Temperature)
V (Exhaust Vacuum)
AP (Ambient Pressure)
RH (Relative HumidityPower Output Prediction using Artificial Neural Networks (ANN)
Dataset Link: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

📌 Project Overview

This project builds an Artificial Neural Network (ANN) model to predict the power output of a Combined Cycle Power Plant (CCPP) based on various environmental factors. The model is trained on a dataset containing power plant performance data and uses deep learning techniques for regression.

📊 Dataset: Combined Cycle Power Plant (CCPP)

Source: UCI Machine Learning Repository

File: Folds5x2_pp.xlsx

Features (Inputs):

AT → Ambient Temperature (°C)

V → Exhaust Vacuum (cm Hg)

AP → Ambient Pressure (millibar)

RH → Relative Humidity (%)

Target Variable (Output):

PE → Net hourly electrical energy output (MW)

🏗 Model Architecture

The ANN Model consists of:

Input Layer: 4 neurons (one for each feature)

Hidden Layers:

Layer 1: 12 neurons, ReLU activation

Layer 2: 8 neurons, ReLU activation

Output Layer: 1 neuron (Predicting continuous values, linear activation)

🔧 Installation & Setup

Clone the repository:

git clone https://github.com/your-repo/PowerOutput-ANN.git
cd PowerOutput-ANN

Install dependencies:

pip install numpy pandas tensorflow matplotlib scikit-learn seaborn

Run the Jupyter Notebook:

jupyter notebook Artificial_Neural_Network.ipynb

🏋️‍♂️ Model Training & Evaluation

Preprocess Data

Load the dataset.

Scale features using StandardScaler.

Train the ANN Model

Using adam optimizer and mean_squared_error loss function.

Train for 100 epochs with batch size 32.

Evaluate Performance

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

Visualize Results

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Power Output')
plt.ylabel('Predicted Power Output')
plt.legend()
plt.show()

🏆 Results & Observations

The model achieves a low RMSE and high R² score, indicating accurate predictions.

The Residuals plot shows a near-normal distribution, suggesting minimal bias.

Possible Improvements:

Experiment with different optimizers (SGD, RMSprop).

Increase model complexity (add layers or neurons).

Use dropout regularization to prevent overfitting.

💾 Saving & Loading the Model

To avoid retraining, you can save and reload the model:

ann.save("power_output_model.h5")
loaded_model = tf.keras.models.load_model("power_output_model.h5")
y_pred = loaded_model.predict(X_test_scaled)

📢 Contributing

Feel free to contribute by improving the model, optimizing performance, or adding new features!

📜 License

MIT License. See LICENSE for details.

🚀 Developed by Your Name | Contact: your.email@example.com


The dataset consists of 9,568 rows and 5 columns (4 features + target variable PE).

