# Bitcoin Price Prediction Project 
---

## 🚀 Project Vision: Toward a Robo-Advisory Platform

This Bitcoin prediction project is a stepping stone toward a larger vision I've been pursuing through my earlier personal finance app — [Dinaromall](https://sepehrpg.ir/portfolio/dinaromall).

The ultimate goal is to build an AI-powered personal financial assistant — a Robo-Adviser that empowers users to:

- Manage their day-to-day finances
- Receive data-driven investment insights
- Predict market trends in crypto and beyond
- Make informed, personalized investment decisions

This is why I chose Bitcoin price forecasting for this machine learning project. It serves as a proof-of-concept for integrating predictive models into a larger intelligent platform that can one day support financial well-being and investment planning for everyday users.

While my current knowledge in areas like model tuning, feature engineering, and data pipelines is still developing, each step I take is bringing me closer to that vision.




## Overview
The core implementation of this Bitcoin price prediction project is available in the [BtcPrediction Notebook](https://github.com/sepehrpg/cryptoPrediction/blob/main/BtcPrediction.ipynb) and modular Python files within this repository. It combines 3 machine learning models into a unified, extensible pipeline:
- Linear Regression
- XGBoost
- LSTM (Long Short-Term Memory)

The dataset is collected using the Yahoo Finance API (`yfinance`).

The project follows a modular structure, with each component such as data collection, preprocessing, modeling, and evaluation defined in separate Python scripts.

---

## 🔧 Data Collection

**File:** `data_collection_yfinance.py`

- Uses `yfinance.download()` to fetch data for BTC-USD.
- Cleans MultiIndex columns.
- Keeps relevant columns: `Open`, `High`, `Low`, `Close`, `Volume`.


---

## 📈 Data Preprocessing

**File:** `data_preprocessing.py`



Features added:

- Lagged features (e.g., `Close_lag1`)
- Technical indicators (e.g., `SMA`, `RSI`, `Bollinger Bands`)
- Date-time features (`Year`, `Month`, `Day`, `DayOfWeek`)
- Optional MinMax scaling

Additional utility functions:

- `add_datetime_features(df)`
- `add_technical_indicators(df)`
- `scale_data(df, columns)`

---

## 📊 Data Splitting

**File:** `data_split.py`



- Chronological train/test split (default 80/20)

---

## 🤖 Models Used

### 1. Linear Regression

**File:** `linear_regression_model.py`

- Handles both scaled and unscaled data
- Automatically inverse transforms using the `MinMaxScaler`

**Evaluation (Unscaled):**

- RMSE: \$2,219.67
- MAE:  \$1,616.53
- R²:   0.9314 ✅

**Notes:**

Although **Linear Regression** showed the best performance based on R² and error metrics, this result is misleading. This model captures only simple linear patterns and assumes independence between features. In highly volatile markets like cryptocurrency, such assumptions break down quickly. The model's good performance is likely due to **overfitting on historical trends** or **smooth market segments**.

In real-world scenarios, Linear Regression generally fails to capture:

- Non-linear price behaviors
- Market sentiment and news impact
- High volatility and abrupt jumps

> ✅ *Conclusion:* Not suitable for real-world crypto forecasting despite seemingly strong metrics in a controlled experiment.

---

### 2. XGBoost

**File:** `xgboost_model.py`

- Powerful gradient boosting model
- Fully tunable

**Evaluation:**

- RMSE: \$6,131.93
- MAE:  \$4,908.51
- R²:   0.4762 ❌

**Notes:**

XGBoost is widely used in **real-world financial and time series applications**, and it's known for its **robustness and performance**. However, in this project, it performed poorly.

Possible reasons:

- Inadequate **hyperparameter tuning**
- Insufficient **feature richness** (XGBoost needs strong features)
- Possibly **data leakage** or **misaligned data**

Despite efforts to improve the model (as noted during project development), the current results do not reflect XGBoost's potential.
For a deeper understanding of the project’s limitations, see the [Personal Reflection](#-personal-reflection) section below.

> ✅ *Conclusion:* While current performance is weak, XGBoost **can be a strong candidate** in real projects — but only when properly tuned with rich feature sets.

---

### 3. LSTM (Long Short-Term Memory)

**File:** `lstm_model.py`

- Deep learning model using `Keras`
- Uses sequences of `time_steps=60`
- Handles reshaping, training, evaluation

**Evaluation:**

- RMSE: \$7,563.33
- MAE:  \$6,195.27
- R²:   0.8165 ✅

**Notes:**

LSTM models are explicitly designed for **sequential data** and **time-series forecasting**. They maintain memory over past observations, making them highly suited for **financial data with temporal dependencies**.

Advantages:

- Captures temporal patterns
- Handles sequences and lagged dependencies
- Scalable with deeper architectures (e.g., stacked LSTM, Bidirectional LSTM)

Opportunities for improvement:

- Longer training and more epochs with regularization
- Use of **external data** (sentiment, macroeconomic indicators)
- Applying **hybrid models** (e.g., ARIMA + LSTM or CNN-LSTM)

> ✅ *Conclusion:* Best candidate for this problem type. With more data and advanced tuning, LSTM would likely outperform all others.

---

## 📌 Model Evaluation Metrics

Common metrics across models:

- **RMSE**: Root Mean Squared Error

  - Python: `np.sqrt(mean_squared_error(y_true, y_pred))`
  - \(\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}\)

- **MAE**: Mean Absolute Error

  - Python: `mean_absolute_error(y_true, y_pred)`
  - \(\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|\)

- **R²**: Coefficient of Determination

  - Python: `r2_score(y_true, y_pred)`
  - \(R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\)

---

## 📌 Conclusion
- **Not Practical For Live Market Scenarios.**: Linear Regression
- **Most Practical (Real-World)**: LSTM — captures time dependencies, scalable.
- **Underperformer**: XGBoost — requires better tuning and features.

---


## 📜 Personal Reflection

As someone who is relatively new to the field of machine learning and time-series forecasting, I openly acknowledge that my **lack of deep knowledge in several key areas** has affected the final performance of this project.

### Areas where my understanding was limited include:

- 🔍 Effective **feature engineering and selection**
- ⚙️ Proper **hyperparameter tuning**
- 🧪 Utilizing appropriate **time-series validation strategies**
- 🧼 Advanced **data preprocessing** techniques
- 🔄 Handling **sequence data** optimally (especially for LSTM)
- 📊 Understanding the impact of **data leakage** or misalignment

---

This project is just the beginning. In the future, I plan to:

- Enhance models with sentiment analysis on financial news.
- Add advanced models like CNN-LSTM and hybrid methods.
- Use ensemble techniques to boost prediction accuracy.
- Apply improved pipelines to other financial forecasting tasks.
- Expand my knowledge in crypto markets and blockchain analytics.

