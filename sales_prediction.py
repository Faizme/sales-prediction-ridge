## Step 1: Import Necessary Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

## Step 2: Data Loading
data = pd.read_csv('datasets/Sales.csv')

## Step 3: Data Exploration
print(data.head())
print(data.info())
print(data.describe())

## Step 4: Data Preprocessing
data.fillna(data.mean(), inplace=True)

## Step 5: Feature Selection
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']  # Target variable

## Step 6: Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0)  # Using Ridge Regression
model.fit(X_train, y_train)

## Step 7: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)

## Step 8: Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred[residuals >= 0], y=residuals[residuals >= 0], color='blue', label='Positive Residuals')
sns.scatterplot(x=y_pred[residuals < 0], y=residuals[residuals < 0], color='orange', label='Negative Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sales')
plt.grid(True)
plt.legend()
plt.show()

## Step 9: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test, color='blue', label='Actual Sales')  # Actual sales
sns.scatterplot(x=y_test, y=y_pred, color='orange', label='Predicted Sales')  # Predicted sales
plt.plot(y_test, y_test, color='red', linewidth=2, label='Regression Line')  # Adding a regression line
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)  # Adding a grid for better readability
plt.legend()
plt.show()