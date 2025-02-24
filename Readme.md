# Power Output Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project builds an **Artificial Neural Network (ANN)** model to predict the **power output** of a **Combined Cycle Power Plant (CCPP)** based on various environmental factors. The model is trained on a dataset containing power plant performance data and uses **deep learning techniques** for regression.

## 📊 Dataset: Combined Cycle Power Plant (CCPP)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
- **File:** `Folds5x2_pp.xlsx`
- **Features (Inputs)**:
  - `AT` → Ambient Temperature (°C)
  - `V` → Exhaust Vacuum (cm Hg)
  - `AP` → Ambient Pressure (millibar)
  - `RH` → Relative Humidity (%)
- **Target Variable (Output):**
  - `PE` → Net hourly electrical energy output (MW)

## 🏗 Model Architecture
The **ANN Model** consists of:
- **Input Layer:** 4 neurons (one for each feature)
- **Hidden Layers:**
  - Layer 1: 12 neurons, **ReLU** activation
  - Layer 2: 8 neurons, **ReLU** activation
- **Output Layer:** 1 neuron (Predicting continuous values, **linear activation**)

## 🔧 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/PowerOutput-ANN.git
   cd PowerOutput-ANN
2. **Install dependencies:**
  ```bash
   pip install numpy pandas tensorflow matplotlib scikit-learn seaborn

3. 
