### Task 1 - Exploration of Customer Purchasing Behavior

## Project Overview

### 1. Repository Setup
- Create a GitHub repository with a dedicated branch: `feature/task-1`.
- Commit progress at least three times a day with descriptive messages.

### 2. Exploratory Data Analysis (EDA)
The objective of this phase is to provide a comprehensive analysis of customer behavior across various stores. Key tasks include:

#### Data Cleaning:
- **Data Quality Assessment**: Check for missing values.
- **Outlier Handling**: Build pipelines to detect and handle outliers.
- **Distribution Analysis**: Check for distribution consistency in both training and test datasets.

#### Behavioral Insights:
- Analyze sales behavior **before**, **during**, and **after holidays**.
- Identify any **seasonal purchasing behaviors** (e.g., Christmas, Easter).
- Explore the correlation between **sales** and the **number of customers**.
- Assess the impact of **promotions** on sales and customer visits.
- Analyze **customer behavior trends** during store opening and closing times.
- Investigate how the **assortment type** affects sales.
- Examine the effect of **distance to competitors** on sales.
- Assess the impact of **new competitor openings or reopenings** on store performance.

#### How to Use
Clone the Repository: git clone https://github.com/hanaDemma/Pharmaceuticals-finance-data-Analysis-week-4.

    Switch to Task-1 Branch:

    git checkout task-1

    Run the Notebook:

    Install dependencies:

    pip install -r requirements.txt

    Open and execute the Jupyter notebook.

### Task 1.2 - Logging
The goal is to log all steps using the `logging` library in Python for traceability and reproducibility.

#### Logging Best Practices:
- Log the start and end of major processes.
- Include checkpoints for critical EDA steps (e.g., missing value checks, outlier detection).
- Log summary statistics and key findings at each stage.
- Ensure logs are written to a file for persistent traceability.

### Task 2 - Prediction of Store Sales
The  goal is to predict daily sales in various stores up to 6 weeks ahead of time

#### Preprocessing
The goal is to prepare the dataset for machine learning models. Key tasks include:

**Handling Missing**:  Values: Impute missing values where necessary.
**Feature Engineering**: Extract relevant features (e.g., weekday, weekends, holiday proximity).
**Data Scaling**: Apply scaling methods (e.g., StandardScaler) to prepare the dataset for machine learning models.

#### Building Machine Learning Models
In this phase, machine learning models will be developed to predict store sales six weeks ahead.

**Algorithms**: Start with tree-based models like Random Forest Regressor.
**Sklearn Pipelines**: Create modular and reusable pipelines for preprocessing and modeling.

#### Choosing a Loss Function
Objective: Select an appropriate loss function to evaluate the model's performance.

#### Post-Prediction Analysis
**Feature Importance**: Analyze which features contributed most to the predictions.
**Confidence Intervals**: Estimate confidence intervals for the predictions to assess reliability.

#### Model Serialization
Objective: Save trained models with timestamps for future use. (e.g., model-2025-01-14-16-30.pkl).
### Deep Learning for Sales Prediction
#### Data Preparation for Time Series
**Time Series Formatting**: Convert the sales data into time series format.
**Stationarity**: Check if the time series is stationary; if not, apply differencing or other transformations.

#### Building an LSTM Model
**Framework**: Use TensorFlow or PyTorch for building the LSTM model.
**Model Architecture**: Start with a simple two-layer LSTM to predict future sales.
**Training**: Train the model on scaled time-series data.

#### Model Evaluation
**Evaluation Metrics**: Use appropriate metrics like MSE or RMSE to evaluate model performance.


### Development Instructions
- Create a feature/task-2 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

### Task 3 - Serving Predictions with a REST API

#### API Development
**Framework**: Choose a framework like Flask or FastAPI for the REST API.
**Endpoints**: Create endpoints that accept input data (e.g., store, date) and return sales predictions.

#### Model Integration
Objective: Load the serialized machine learning models from Task 2 to generate predictions.

#### Deployment
Objective: Deploy the API to a web server or cloud platform for real-time usage.


### Development Instructions
- Create a task-3 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).