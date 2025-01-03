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

def label_holiday_periods(row, holiday_type):
    # logger.info(f"Labeling holiday periods for {holiday_type}.")
    if row['StateHoliday'] == holiday_type:
        return 'During ' + holiday_type
    else:
        return 'Regular'

def categorizeEachDayBasedonHolidayType(train_data):
    logger.info("Categorizing each day based on holiday type.")
    for holiday in ['Public Holiday', 'Easter Holiday', 'Christmas']:
        # Apply the holiday categorization
        train_data[f'HolidayPeriod_{holiday}'] = train_data.apply(lambda row: label_holiday_periods(row, holiday), axis=1)
        
        # Create columns to indicate "Before" and "After" holiday periods, adding checks to ensure we have data
        train_data[f'AfterHoliday_{holiday}'] = train_data[f'HolidayPeriod_{holiday}'].shift(-1) == f'During {holiday}'
        train_data[f'BeforeHoliday_{holiday}'] = train_data[f'HolidayPeriod_{holiday}'].shift(1) == f'During {holiday}'
        
        # Initialize with "Regular" and then adjust for "Before" and "After" periods
        train_data[f'SalesPeriod_{holiday}'] = 'Regular'
        train_data.loc[train_data[f'BeforeHoliday_{holiday}'], f'SalesPeriod_{holiday}'] = f'Before {holiday}'
        train_data.loc[train_data[f'HolidayPeriod_{holiday}'] == f'During {holiday}', f'SalesPeriod_{holiday}'] = f'During {holiday}'
        train_data.loc[train_data[f'AfterHoliday_{holiday}'], f'SalesPeriod_{holiday}'] = f'After {holiday}'
    return train_data

def calculateAverageSalesDuringDifferentPeriods(train_data):
    logger.info("Calculating average sales during different periods.")
    # Check if there are sales during holidays; if empty, no need to plot
    holiday_sales_behavior = pd.DataFrame()
    for holiday in ['Public Holiday', 'Easter Holiday', 'Christmas']:
        # Filter out only the relevant holiday sales periods
        filtered_sales = train_data[train_data[f'SalesPeriod_{holiday}'] != 'Regular']
        
        # Ensure there's data before proceeding with analysis
        if not filtered_sales.empty:
            # Calculate average sales during different periods (Before, During, After)
            sales_behavior = filtered_sales.groupby(f'SalesPeriod_{holiday}')['Sales'].mean()
            sales_behavior.name = holiday
            holiday_sales_behavior = pd.concat([holiday_sales_behavior, sales_behavior], axis=1)
        else:
            print(f"No data found for {holiday}, skipping...")

    order = [
        'Before Public Holiday', 'During Public Holiday', 'After Public Holiday',
        'Before Easter Holiday', 'During Easter Holiday', 'After Easter Holiday',
        'Before Christmas', 'During Christmas', 'After Christmas'
    ]

    # Reindex the holiday_sales_behavior DataFrame to follow the desired order
    holiday_sales_behavior = holiday_sales_behavior.reindex(order)
    return holiday_sales_behavior


def plotEffectOfHolidayOnSales(holiday_sales_behavior):
    logger.info("Plotting the effect of holidays on sales.")
    if not holiday_sales_behavior.empty:
        holiday_sales_behavior.plot(kind='bar', figsize=(14, 7), colormap='plasma')
        
        # Add labels and title
        plt.title('Sales Behavior Before, During, and After Different Holidays')
        plt.ylabel('Average Sales')
        plt.xlabel('Holiday Period')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
        plt.tight_layout()
        
        # Show plot
        plt.show()
    else:
        print("No holiday sales data to plot.")



def salesOverTime(merged_train_data_store):
    logger.info("Plotting sales over time by different timeframes.")
    for column in ['D','W','M','Y']:
        over_time_sales = merged_train_data_store['Sales'].resample(column).sum()
        plt.figure(figsize=(15, 7))
        plt.plot(over_time_sales.index, over_time_sales)
        if column =='D':
            plt.title('Daily Sales Over Time')
        elif column =='W':
            plt.title('Weekly Sales Over Time')
        elif column =='M':
            plt.title('Monthly Sales Over Time')
        else:
            plt.title('Yearly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()



def salesSeasonalDecompose(merged_train_data_store):
    logger.info("Decomposing sales seasonality.")
    monthly_sales = merged_train_data_store['Sales'].resample('M').sum()
    result = seasonal_decompose(monthly_sales, model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 5)  

    plt.tight_layout() 
    plt.show()


def averageSalesDayOfWeek(merged_train_data_store):
    logger.info("Calculating average sales by day of the week.")
    merged_train_data_store['DayOfWeek'] = merged_train_data_store.index.dayofweek
    day_of_week_sales = merged_train_data_store.groupby('DayOfWeek')['Sales'].mean()
    day_of_week_sales.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Sales')
    plt.show()


def salesWithOpenAndClose(monthly_open_store):
    logger.info("Plotting monthly sales: Open vs Not Open.")
    monthly_open_store[[0, 1]].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Open vs Not Open')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['Not Open', 'Open'])
    plt.show()


def customerBehaviorStoreOpen(train_data):
    logger.info("Analyzing customer behavior for stores that are open.")
    open_data = train_data[train_data['Open'] == 1]
    monthly_customers = open_data.groupby('Month')['Customers'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_customers.index, monthly_customers.values, marker='o', linestyle='-', color='b')
    plt.title('Average Number of Customers by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Customers')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Month names
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def customerBehaviorStoreNotOpen(train_data):
    logger.info("Analyzing customer behavior for stores that are not open.")
    train_data['Date'] = pd.to_datetime(train_data['Date'])


    train_data['Month'] = train_data['Date'].dt.month

    open_data = train_data[train_data['Open'] == 0]

    monthly_customers = open_data.groupby('Month')['Customers'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_customers.index, monthly_customers.values, marker='o', linestyle='-', color='b')
    plt.title('Average Number of Customers by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Customers')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Month names
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def promotionEffectSales(monthly_promo_sales):
    logger.info("Plotting effect of promotions on monthly average sales.")
    monthly_promo_sales[[0, 1]].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()

def storeTypePerformanceOverTime(merged_train_data_store):
    logger.info("Analyzing store type performance over time.")
    store_type_sales = merged_train_data_store.groupby([merged_train_data_store.index.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()
    return store_type_sales