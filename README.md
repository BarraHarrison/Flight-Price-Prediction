# Flight Price Prediction Model ✈️💰

## Introduction 📝
Predicting flight prices is a challenging yet valuable task in the airline industry. By leveraging machine learning techniques, the aim is to build a predictive model that estimates the price of flights based on various features such as airline, departure time, arrival time, source city, destination city, and travel class. This project implements a **Random Forest Regression model** to analyze and predict flight prices, improving decision-making for travelers and airline companies alike.

---

## Data PreProcessing 🔄
### Purpose of Data PreProcessing
Data preprocessing is a crucial step in machine learning that involves cleaning and transforming raw data into a format suitable for model training. This ensures that the dataset is free from unnecessary columns, properly formatted, and ready for effective learning.

#### Key Steps:
- **Dropping Unnecessary Columns**: The `Unnamed: 0` and `flight` columns are removed as they do not contribute to predicting flight prices.
  ```python
  data_frame = data_frame.drop("Unnamed: 0", axis=1)
  data_frame = data_frame.drop("flight", axis=1)
  ```
- **Encoding Categorical Variables**: The `class` column is converted into a binary format where "Business" class is represented as `1` and other classes as `0`.
  ```python
  data_frame["class"] = data_frame["class"].apply(lambda x: 1 if x == "Business" else 0)
  ```

---

## Handling Categorical Features with pandas "get_dummies" 📊
### Purpose of `get_dummies`
Many machine learning models require numerical input. Since categorical columns like `airline`, `source_city`, and `destination_city` contain text values, I converted them into numerical format using **one-hot encoding** with `pandas.get_dummies()`.

#### Key Transformations:
- Convert categorical columns into separate binary columns:
  ```python
  data_frame.join(pd.get_dummies(data_frame.airline, prefix="airline")).drop("airline", axis=1)
  data_frame.join(pd.get_dummies(data_frame.source_city, prefix="source")).drop("source_city", axis=1)
  data_frame.join(pd.get_dummies(data_frame.destination_city, prefix="dest")).drop("destination_city", axis=1)
  data_frame.join(pd.get_dummies(data_frame.arrival_time, prefix="arrival")).drop("arrival_time", axis=1)
  data_frame.join(pd.get_dummies(data_frame.departure_time, prefix="departure")).drop("departure_time", axis=1)
  ```
- This ensures that categorical variables are transformed into multiple binary features, making them usable for the regression model.

---

## Training a Regression Model 🤖📈
### Purpose of Training a Regression Model
Regression models predict continuous values, such as flight prices, based on input features. The **Random Forest Regressor** is chosen because it is a powerful ensemble learning technique that reduces overfitting and improves accuracy.

#### Key Steps:
1. **Import necessary libraries**:
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestRegressor
   ```
2. **Check the dataset**:
   ```python
   print(type(data_frame))
   print(data_frame.head())
   ```
3. **Train the model**:
   ```python
   reg = RandomForestRegressor(n_jobs=-1)
   reg.fit(x_train, y_train)
   ```
   - The `RandomForestRegressor` is trained using the `fit()` function, where it learns patterns from the training data.

---

## Model Evaluation with Sklearn Metrics 📊✅
### Purpose of Sklearn Metrics
To measure the performance of the regression model, I used multiple evaluation metrics:
- **R² Score (R-squared)**: Indicates how well the model explains the variance in the target variable.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted prices.
- **Mean Squared Error (MSE)**: Calculates the average squared difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an error measure in the original unit.

#### Implementation:
```python
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_predicted = reg.predict(x_test)
print("R2: ", r2_score(y_test, y_predicted))
print("MAE: ", mean_absolute_error(y_test, y_predicted))
print("MSE: ", mean_squared_error(y_test, y_predicted))
print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_predicted)))
```
These metrics help assess how well the model is performing in predicting flight prices.

---

## Visualizing Predictions with a Scatterplot 📊📌
### Purpose of Scatterplot with Matplotlib
A scatterplot is used to compare actual flight prices against the model’s predicted values. This visualization helps identify patterns, potential biases, or overfitting.

#### Implementation:
```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_predicted)
plt.xlabel("Actual Flight Price")
plt.ylabel("Predicted Flight Price")
plt.title("Prediction VS Actual Price")
plt.show()
```
- If the model performs well, points should be close to the **y = x** diagonal, indicating accurate predictions.

---

## Conclusion 🎯
This project successfully builds a **Random Forest Regression model** to predict flight prices based on various factors. By performing **data preprocessing**, **one-hot encoding categorical features**, **training a regression model**, **evaluating performance with metrics**, and **visualizing results**,  insights are gained in relation to pricing patterns. This approach can help both travelers and airline companies make informed decisions.

🚀 Future improvements could include using more advanced models like **XGBoost** or **Neural Networks**, and incorporating **real-time pricing data** to improve accuracy! 🌍

