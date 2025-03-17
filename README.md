# Sales Prediction using Ridge Regression

This project predicts sales based on advertising spend using Ridge Regression. The model is trained and evaluated using Python's `scikit-learn` library.

## Overview
The goal of this project is to predict sales based on the advertising budget allocated to TV, radio, and newspapers. Ridge Regression is used to prevent overfitting and improve model performance.

## Dataset
- **File:** `Sales.csv`
- **Features:**
  - `TV`: Advertising spend on TV
  - `Radio`: Advertising spend on Radio
  - `Newspaper`: Advertising spend on Newspapers
- **Target:** `Sales`

## Installation
Clone the repository:
```bash
git clone https://github.com/Faizme/sales-prediction-ridge.git
cd sales-prediction-ridge
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Jupyter Notebook
```bash
jupyter notebook sales_prediction.ipynb
```

### Running the Python Script
```bash
python sales_prediction.py
```

## Project Structure
```
├── datasets
│   └── Sales.csv
├── sales_prediction.ipynb
├── sales_prediction.py
├── requirements.txt
└── README.md
```

## Results
### Model Performance
- **Mean Squared Error (MSE):** Measures the average squared difference between the actual and predicted sales.
- **R-squared:** Measures how well the model explains the variance in sales data.

### Plots
1. **Residual Plot:** Shows the residuals (actual - predicted) to assess the fit.
2. **Actual vs Predicted Sales:** Displays how well the predicted values match the actual values.

## Contributing
Feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.

