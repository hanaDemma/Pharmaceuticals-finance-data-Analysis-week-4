
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from datetime import datetime

# Model Building
def build_ml_model():
    """
    Build a Random Forest Regressor with 50 estimators and random_state for reproducibility.
    """
    model = RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1)
    return model

# Model Training with Sklearn Pipeline
def train_ml_model(train_data):
    train_data_model = train_data
    train_data_no_sales = train_data_model.drop('Sales',axis=1)
    X_train = train_data_no_sales
    y_train = train_data.pop('Sales')

    
    # Defining the pipeline
    pipeline = Pipeline([
        ('model', build_ml_model())
    ])
    
    # Fitting the model
    pipeline.fit(X_train, y_train)
    
    # Making predictions on validation data
    predictions = pipeline.predict(X_train)
    
    # Calculating evaluation metrics
    rmse = mean_squared_error(y_train, predictions)
    mae = mean_absolute_error(y_train, predictions)
    
    print(f"Validation RMSE: {rmse}")
    print(f"Validation MAE: {mae}")
   
    
    return pipeline,X_train,y_train


def post_prediction_feature_importance(model, train_data):
    """
    Analyze feature importance and estimate confidence intervals for predictions.
    """
    # Extracting feature importance from the Random Forest model
    model_rf = model.named_steps['model']

    feature_importances = model_rf.feature_importances_

    # Sorting feature importances for visualization
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(train_data.columns)[sorted_idx], feature_importances[sorted_idx], color='b')
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Feature Importance in Sales Prediction")
    plt.show()