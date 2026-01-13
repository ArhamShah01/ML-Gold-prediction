# Gold Price Prediction using LSTM

This project utilizes a deep learning approach to forecast daily gold prices based on historical market data. By leveraging Long Short-Term Memory (LSTM) networks, the model captures temporal dependencies and trends in financial time series data to predict future price movements.

## 📌 Project Overview
The goal of this project is to build a robust regression model that predicts the closing price of gold for the next day using a sequence of past prices (60-day window). The workflow includes comprehensive data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

## 📂 Dataset
The model is trained on `GoldPrice.csv`, which contains historical daily gold price data.
- **Features used:** `Date`, `Price`
- **Preprocessing:** 
  - Date conversion and sorting.
  - Removal of currency symbols/formatting.
  - Normalization using Min-Max Scaling.

## 🛠 Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras (LSTM, Dense, Dropout layers)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Plotly
- **Machine Learning Utilities:** Scikit-learn (MinMaxScaler, train_test_split, metrics)

## 🧠 Model Architecture
The neural network consists of a stacked LSTM architecture designed to prevent overfitting while learning complex patterns:
1.  **Input Layer:** Sequence length of 60 days.
2.  **LSTM Layer 1:** 64 units, returns sequences, followed by Dropout (0.2).
3.  **LSTM Layer 2:** 64 units, returns sequences, followed by Dropout (0.2).
4.  **LSTM Layer 3:** 64 units, followed by Dropout (0.2).
5.  **Dense Layer:** 32 units with Softmax activation.
6.  **Output Layer:** 1 unit (linear) for price regression.

**Optimizer:** Nadam  
**Loss Function:** Mean Squared Error (MSE)

## 📊 Performance & Evaluation
- The dataset is split into training and testing sets (Test set: Year 2022 data).
- **Metric:** Mean Absolute Percentage Error (MAPE) and Accuracy.
- **Visualization:** The project generates a final plot comparing:
  - Historical Training Data
  - Actual Test Prices (Ground Truth)
  - Predicted Test Prices

## 🚀 How to Run
1.  Ensure `GoldPrice.csv` is in the same directory as the notebook.
2.  Install required dependencies:
    ```bash
    pip install pandas numpy matplotlib plotly scikit-learn tensorflow
    ```
3.  Run the Jupyter Notebook `gold_price_lstm.ipynb`.

## 📈 Results
The final output visualizes the model's ability to track market trends, displaying the trajectory of predicted prices against actual market values.
