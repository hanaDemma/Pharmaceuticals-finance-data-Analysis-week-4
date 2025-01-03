import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scripts.logging import logger
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def replace_char_state_holiday(char):
   
    if char == 'a':
        return 'Public Holiday'
    elif char == 'b':
        return 'Easter Holiday'
    elif char == 'c':
        return 'Christmas'
    else:
        return 'Non Holiday'
def replace_char_assortment(char):
    # logger.info(f"Replacing char '{char}' for assortment.")
    if char == 'a':
        return 'basic'
    elif char == 'b':
        return 'extra'
    elif char == 'c':
        return 'extended'
    else:
        return char
    
def create_date_features(df):
    logger.info("Creating date features.")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df
def month_period(day):
    if day <= 10:
        return 'beginning'
    elif day <= 20:
        return 'middle'
    else:
        return 'end'
    

def distribution_promotions_in_both_datasets(merged_train_data_store,merged_test_data_store):
    logger.info("Comparing Promo distribution between train and test datasets.")
    train_promo_dist = merged_train_data_store['Promo'].value_counts(normalize=True) * 100
    test_promo_dist = merged_test_data_store['Promo'].value_counts(normalize=True) * 100

    # Create a DataFrame to compare the distributions
    promo_comparison = pd.DataFrame({
        'Train': train_promo_dist,
        'Test': test_promo_dist
    }).transpose()

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    sns.barplot(x=promo_comparison.columns, y=promo_comparison.loc['Train'], color='purple', label='Train')
    sns.barplot(x=promo_comparison.columns, y=promo_comparison.loc['Test'], color='red',alpha=0.3, label='Test')

    # Add labels and title
    plt.title('Comparison of Promo Distribution in Training and Test Sets')
    plt.ylabel('Percentage of Promo Days')
    plt.legend()

    plt.show()