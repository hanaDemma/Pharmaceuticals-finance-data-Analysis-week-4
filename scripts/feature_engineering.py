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
    """
    Compare the Promo distribution between train and test datasets and visualize.
    """
    logger.info("Comparing Promo distribution between train and test datasets.")
    train_promo_dist = merged_train_data_store['Promo'].value_counts(normalize=True) * 100
    test_promo_dist = merged_test_data_store['Promo'].value_counts(normalize=True) * 100

   
    promo_comparison = pd.DataFrame({
        'Train': train_promo_dist,
        'Test': test_promo_dist
    }).transpose()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=promo_comparison.columns, y=promo_comparison.loc['Train'], color='purple', label='Train')
    sns.barplot(x=promo_comparison.columns, y=promo_comparison.loc['Test'], color='red',alpha=0.3, label='Test')

    plt.title('Comparison of Promo Distribution in Training and Test Sets')
    plt.ylabel('Percentage of Promo Days')
    plt.legend()

    plt.show()

def label_holiday_periods(row, holiday_type):
    if row['StateHoliday'] == holiday_type:
        return 'During ' + holiday_type
    else:
        return 'Regular'

def categorizeEachDayBasedonHolidayType(train_data):
    logger.info("Categorizing each day based on holiday type.")
    for holiday in ['Public Holiday', 'Easter Holiday', 'Christmas']:
      
        train_data[f'HolidayPeriod_{holiday}'] = train_data.apply(lambda row: label_holiday_periods(row, holiday), axis=1)
        
        train_data[f'AfterHoliday_{holiday}'] = train_data[f'HolidayPeriod_{holiday}'].shift(-1) == f'During {holiday}'
        train_data[f'BeforeHoliday_{holiday}'] = train_data[f'HolidayPeriod_{holiday}'].shift(1) == f'During {holiday}'
        train_data[f'SalesPeriod_{holiday}'] = 'Regular'
        train_data.loc[train_data[f'BeforeHoliday_{holiday}'], f'SalesPeriod_{holiday}'] = f'Before {holiday}'
        train_data.loc[train_data[f'HolidayPeriod_{holiday}'] == f'During {holiday}', f'SalesPeriod_{holiday}'] = f'During {holiday}'
        train_data.loc[train_data[f'AfterHoliday_{holiday}'], f'SalesPeriod_{holiday}'] = f'After {holiday}'
    return train_data

def calculateAverageSalesDuringDifferentPeriods(train_data):
    """
    Calculate average sales during different periods (Before, During, After) for holidays.
    """
    logger.info("Calculating average sales during different periods.")

    holiday_sales_behavior = pd.DataFrame()
    for holiday in ['Public Holiday', 'Easter Holiday', 'Christmas']:
        filtered_sales = train_data[train_data[f'SalesPeriod_{holiday}'] != 'Regular']
        

        if not filtered_sales.empty:
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
    holiday_sales_behavior = holiday_sales_behavior.reindex(order)
    return holiday_sales_behavior


def plotEffectOfHolidayOnSales(holiday_sales_behavior):
    """
    Plot the effect of holidays on sales based on the provided DataFrame.

    Args:
        holiday_sales_behavior (DataFrame): DataFrame containing average sales during holiday periods.
        save_path (str, optional): File path to save the plot. Defaults to None.
    """
    logger.info("Plotting the effect of holidays on sales.")
    if not holiday_sales_behavior.empty:
        holiday_sales_behavior.plot(kind='bar', figsize=(14, 7), colormap='plasma')
        
      
        plt.title('Sales Behavior Before, During, and After Different Holidays')
        plt.ylabel('Average Sales')
        plt.xlabel('Holiday Period')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
        plt.tight_layout()
    
        plt.show()
    else:
        print("No holiday sales data to plot.")



def salesOverTime(merged_train_data_store):
    """
    Plot sales over time for specified timeframes.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing 'Sales' column and a DateTimeIndex.
        timeframes (list): List of timeframes to resample by (e.g., ['D', 'W', 'M', 'Y']).
    """
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

    """
    Decompose sales data into trend, seasonal, and residual components.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing 'Sales' column with a DateTimeIndex.
        freq (str): Resampling frequency ('M' for monthly, 'W' for weekly, etc.).
        model (str): Decomposition model ('additive' or 'multiplicative').

    Returns:
        result (DecomposeResult): The decomposition result object.
    """

    logger.info("Decomposing sales seasonality.")
    monthly_sales = merged_train_data_store['Sales'].resample('M').sum()
    result = seasonal_decompose(monthly_sales, model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 5)  

    plt.tight_layout() 
    plt.show()


def averageSalesDayOfWeek(merged_train_data_store):
    """
    Calculate and plot the average sales by day of the week.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing 'Sales' column with a DateTimeIndex.
    """
    logger.info("Calculating average sales by day of the week.")
    merged_train_data_store['DayOfWeek'] = merged_train_data_store.index.dayofweek
    day_of_week_sales = merged_train_data_store.groupby('DayOfWeek')['Sales'].mean()
    day_of_week_sales.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Sales')
    plt.show()


def salesWithOpenAndClose(monthly_open_store):
    """
    Plot monthly average sales for stores when they are open versus not open.

    Args:
        monthly_open_store (DataFrame): DataFrame with average sales for 'Open' and 'Not Open' stores.
    """
    logger.info("Plotting monthly sales: Open vs Not Open.")
    monthly_open_store[[0, 1]].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Open vs Not Open')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['Not Open', 'Open'])
    plt.show()


def customerBehaviorStoreOpen(train_data):
    """
    Analyze and plot the average number of customers for stores that are open by month.

    Args:
        train_data (DataFrame): DataFrame containing 'Open', 'Customers', and date-related information.
    """
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
    """
    Analyze and plot the average number of customers for stores that are not open by month.

    Args:
        train_data (DataFrame): DataFrame containing 'Open', 'Customers', and 'Date' columns.
    """
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
    """
    Plot the effect of promotions on monthly average sales.

    Args:
        monthly_promo_sales (DataFrame): DataFrame with monthly average sales for 'Promo' and 'No Promo'.
    """
    logger.info("Plotting effect of promotions on monthly average sales.")
    monthly_promo_sales[[0, 1]].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()

def storeTypePerformanceOverTime(merged_train_data_store):
    """
    Analyze and plot the monthly average sales by store type over time.

    Args:
        merged_train_data_store (DataFrame): DataFrame with sales data and a store type column.
    
    Returns:
        DataFrame: Monthly average sales for each store type.
    """
    logger.info("Analyzing store type performance over time.")
    store_type_sales = merged_train_data_store.groupby([merged_train_data_store.index.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()
    return store_type_sales


def storeTypeAndPromoOverTime(merged_store_type_prome):
    """
    Plot the performance of sales by store type and promotion status over time.

    Args:
        merged_store_type_prome (DataFrame): DataFrame containing sales data with store types and promotions.
    
    Returns:
        None
    """
    logger.info("Plotting performance by store type and promotion over time.")
    merged_store_type_prome.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type and Promotion')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.legend(['No Promo', 'Promo','Store A', 'Store B', 'Store C', 'Store D'])

    plt.show()

def numberOfCustomerWithSales(merged_train_data_store):
    """
    Plot the relationship between the number of customers and sales over time.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing customer and sales data.
    
    Returns:
        None
    """
    logger.info("Plotting the relationship between number of customers and sales over time.")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(merged_train_data_store['Customers'], merged_train_data_store['Sales'], c=merged_train_data_store.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time ')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

def corrSalesByDayOfWeekAndMonth(merged_train_data_store):
    """
    Create a heatmap for average sales by day of the week and month.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing sales data indexed by datetime.
    
    Returns:
        None
    """
    logger.info("Creating a heatmap for average sales by day of the week and month.")
    merged_train_data_store['DayOfWeek'] = merged_train_data_store.index.dayofweek
    merged_train_data_store['Month'] = merged_train_data_store.index.month
    sales_heatmap = merged_train_data_store.pivot_table(values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(sales_heatmap, cmap='coolwarm', annot=True, fmt='.0f')
    plt.title('Average Sales by Day of Week and Month')
    plt.xlabel('Month')
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
    plt.show()

def corrSalesAndCustomers(merged_train_data_store):
    """
    Calculates and visualizes the correlation between Sales and Customers.

    Args:
        merged_train_data_store: A pandas DataFrame containing 'Sales' and 'Customers' columns.

    Returns:
        None. Displays the correlation heatmap.
    """
    logger.info("Calculating and plotting correlation between Sales and Customers.")

    if 'Sales' not in merged_train_data_store.columns or 'Customers' not in merged_train_data_store.columns:
        logger.error("Columns 'Sales' and 'Customers' must be present in the dataset.")
        return
    
    correlation_matrix = merged_train_data_store[['Sales', 'Customers']].corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Between Sales and Customers')
    plt.show()


def dailySalesGrowthRate(merged_train_data_store):
    """
    Plot the daily sales growth rate.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing sales data indexed by datetime.
    
    Returns:
        None
    """
    logger.info("Plotting daily sales growth rate.")
    merged_train_data_store['SalesGrowthRate'] = merged_train_data_store['Sales'].pct_change()
    plt.figure(figsize=(15, 7))
    plt.plot(merged_train_data_store.index, merged_train_data_store['SalesGrowthRate'])
    plt.title('Daily Sales Growth Rate')
    plt.xlabel('Date')
    plt.ylabel('Growth Rate')
    plt.show()


def categorize_month(month):
    """
    Categorize month into a season: Winter, Spring, Summer, Fall.
    
    Args:
        month (int): Month number (1-12).
    
    Returns:
        str: Season name corresponding to the month.
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
    
def customerBehaviorStoreOpenSeasonal(train_data):
    """
    Analyzes customer behavior by season for stores that are open.

    Args:
        train_data (DataFrame): DataFrame containing customer data with a 'Month' and 'Open' column.
    
    Returns:
        None
    """
    logger.info("Analyzing customer behavior by season for stores that are open.")
    train_data['Season'] = train_data['Month'].apply(categorize_month)

    open_data = train_data[train_data['Open'] == 1]

    seasonal_customers = open_data.groupby('Season')['Customers'].mean()

    plt.figure(figsize=(12, 6))
    seasonal_customers.plot(kind='bar', color='b')
    plt.title('Average Number of Customers by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Number of Customers')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def numberStateHolidayAndNotHoliday(merged_train_data_store):
    """
    Plot the distribution of State Holidays in the dataset.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing the 'StateHoliday' column.

    Returns:
        None
    """
    logger.info("Plotting distribution of State Holidays.")
    plt.figure(figsize=(8, 6))
    merged_train_data_store['StateHoliday'].value_counts().plot(kind='bar')
    plt.title('Distribution of state Holiday')
    plt.ylabel("Count")
    plt.show()

def numberSchoolHolidayAndNotHoliday(merged_train_data_store):
    """
    Plot the distribution of School Holidays in the dataset.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing the 'SchoolHoliday' column.

    Returns:
        None
    """
    logger.info("Plotting distribution of School Holidays.")
    plt.figure(figsize=(8, 6))
    merged_train_data_store['SchoolHoliday'].value_counts().plot(kind='bar')
    plt.title('Distribution of School Holiday')
    plt.xlabel('Is Holiday')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'], rotation=0)
    plt.show()


def averageSalesOnStateHoliday(merged_train_data_store):
    """
    Plot the average sales on State Holidays vs Non-Holidays.

    Args:
        merged_train_data_store (DataFrame): DataFrame containing the 'StateHoliday' and 'Sales' columns.

    Returns:
        None
    """
    logger.info("Plotting average sales on State Holidays vs Non-Holidays.")
    holiday_effect = merged_train_data_store.groupby('StateHoliday')['Sales'].mean()
    holiday_effect.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales: State Holiday vs Non-Holiday')
    plt.ylabel('Average Sales')
    plt.show()

def averageSalesOnSchoolHoliday(merged_train_data_store):
    logger.info("Plotting average sales on School Holidays vs Non-Holidays.")
    holiday_effect = merged_train_data_store.groupby('SchoolHoliday')['Sales'].mean()
    holiday_effect.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales: School Holiday vs Non-Holiday')
    plt.ylabel('Average Sales')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'], rotation=0)
    plt.show()


def effectAssortmentTypeOnSales(monthly_effect_assortment_type):
    logger.info("Analyzing the effect of assortment type on monthly average sales.")
    monthly_effect_assortment_type.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()


def effectCompetitionDistanceOnSales(merged_train_data_store):
    logger.info("Analyzing the effect of competition distance on sales.")
    intervals = [0, 5000, 10000, 15000, 20000, 75860.0]
    merged_train_data_store['CompetitionDistanceInterval'] = pd.cut(merged_train_data_store['CompetitionDistance'], bins=intervals)
    interval_sales = merged_train_data_store.groupby('CompetitionDistanceInterval')['Sales'].mean()
    plt.figure(figsize=(10, 6))
    interval_sales.plot(kind='bar')
    plt.title('Average Sales by Competition Distance Interval')
    plt.xlabel('Competition Distance Interval')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.show()



def storesOpenAllWeekdays(merged_train_data_store):
    logger.info("Analyzing stores open on all weekdays.")
    merged_train_data_store['Date'] = pd.to_datetime(merged_train_data_store['Date'])

    open_data = merged_train_data_store[merged_train_data_store['Open'] == 1]

    weekday_open_data = open_data[open_data['DayOfWeek'].isin([0, 1, 2, 3, 4])]

    stores_open_days = weekday_open_data.groupby(['Store', 'StoreType'])['DayOfWeek'].nunique().reset_index()
    stores_open_days['IsOpenAllWeekdays'] = stores_open_days['DayOfWeek'] == 5
    stores_open_all_weekdays = stores_open_days[stores_open_days['IsOpenAllWeekdays'] == True]
    stores_not_open_all_weekdays = stores_open_days[stores_open_days['IsOpenAllWeekdays'] == False]
    open_all_weekdays_by_type = stores_open_all_weekdays.groupby('StoreType').size()
    not_open_all_weekdays_by_type = stores_not_open_all_weekdays.groupby('StoreType').size()

    print("Stores Open on All Weekdays by StoreType:")
    print(open_all_weekdays_by_type)

    print("\nStores NOT Open on All Weekdays by StoreType:")
    print(not_open_all_weekdays_by_type)

    open_weekday_summary = pd.DataFrame({
    'OpenAllWeekdays': open_all_weekdays_by_type,
    'NotOpenAllWeekdays': not_open_all_weekdays_by_type
    }).fillna(0) 
    open_weekday_summary = open_weekday_summary.reset_index()

    plt.figure(figsize=(12, 6))

    open_weekday_summary.set_index('StoreType')[['OpenAllWeekdays', 'NotOpenAllWeekdays']].plot(
        kind='bar', stacked=True, color=['#34a853', '#ea4335'], alpha=0.9, edgecolor='k')

    plt.title('Stores Open All Weekdays vs. Not Open All Weekdays by StoreType', fontsize=14)
    plt.xlabel('Store Type', fontsize=12)
    plt.ylabel('Number of Stores', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(['Open All Weekdays', 'Not Open All Weekdays'], loc='upper right')
    plt.tight_layout()
    plt.show()



def storesOpenWeekdayOpenWeekends(merged_train_data_store):
    logger.info('Starting storesOpenWeekdayOpenWeekends function')
    open_data = merged_train_data_store[merged_train_data_store['Open'] == 1]
    logger.info(f'Filtered open stores: {len(open_data)} entries')

    weekday_data = open_data[open_data['DayOfWeek'].isin([0, 1, 2, 3, 4])]  
    stores_open_weekdays = weekday_data.groupby('Store')['DayOfWeek'].nunique()
    stores_open_all_weekdays = stores_open_weekdays[stores_open_weekdays == 5].index
    logger.info(f'Number of stores open all weekdays: {len(stores_open_all_weekdays)}')

    weekend_data = merged_train_data_store[merged_train_data_store['DayOfWeek'].isin([5, 6])]  
    weekend_sales_open_all_weekdays = weekend_data[weekend_data['Store'].isin(stores_open_all_weekdays)].groupby('Store')['Sales'].mean()
    weekend_sales_not_open_all_weekdays = weekend_data[~weekend_data['Store'].isin(stores_open_all_weekdays)].groupby('Store')['Sales'].mean()
    
    logger.info(f'Average weekend sales for stores open all weekdays: {weekend_sales_open_all_weekdays.mean()}')
    logger.info(f'Average weekend sales for stores NOT open all weekdays: {weekend_sales_not_open_all_weekdays.mean()}')

    comparison_df = pd.DataFrame({
        'Stores Open All Weekdays': [weekend_sales_open_all_weekdays.mean()],
        'Stores Not Open All Weekdays': [weekend_sales_not_open_all_weekdays.mean()]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, palette='coolwarm')
    plt.title('Comparison of Weekend Sales: Stores Open All Weekdays vs. Not Open All Weekdays', fontsize=14)
    plt.ylabel('Average Weekend Sales')
    plt.xlabel('Store Type')
    plt.tight_layout()
    plt.show()

    logger.info('Completed storesOpenWeekdayOpenWeekends function')

def splitHolidayTypeToLists(data):
    data['Date'] = pd.to_datetime(data['Date'])
    # Split holiday types 'a', 'b', 'c' into separate lists
    holiday_a_dates = np.sort(data[data['StateHoliday'] == 'a']['Date'].unique())
    holiday_b_dates = np.sort(data[data['StateHoliday'] == 'b']['Date'].unique())
    holiday_c_dates = np.sort(data[data['StateHoliday'] == 'c']['Date'].unique())
    return holiday_a_dates, holiday_b_dates, holiday_c_dates

def days_to_nearest_holiday(row_date, holidays):
    if holidays.size == 0: 
        return 0
    days_to_holidays = [(holiday - row_date).days for holiday in holidays]
    result = min(days_to_holidays, key=abs)
    if result<0:
        return -1
    else:
        return result
    
def days_after_last_holiday(row_date, holidays):
    past_holidays = [holiday for holiday in holidays if holiday < row_date]
    if past_holidays:
        last_holiday = max(past_holidays)
        result =(row_date - last_holiday).days
        if result<0:
            return -1
        else:
            return result
    else:
        return 0 

def assign_days_to_and_after_holiday(data,holiday_a_dates_train, holiday_b_dates_train, holiday_c_dates_train):
    data['DaysTo_A_Holiday'] = data['Date'].apply(lambda x: days_to_nearest_holiday(x, holiday_a_dates_train))
    data['DaysTo_B_Holiday'] = data['Date'].apply(lambda x: days_to_nearest_holiday(x, holiday_b_dates_train))
    data['DaysTo_C_Holiday'] = data['Date'].apply(lambda x: days_to_nearest_holiday(x, holiday_c_dates_train))
    data['DaysAfter_A_Holiday'] = data['Date'].apply(lambda x: days_after_last_holiday(x, holiday_a_dates_train))
    data['DaysAfter_B_Holiday'] = data['Date'].apply(lambda x: days_after_last_holiday(x, holiday_b_dates_train))
    data['DaysAfter_C_Holiday'] = data['Date'].apply(lambda x: days_after_last_holiday(x, holiday_c_dates_train))
    return data





def get_preprocessed_test_data(test_data):
    # Columns for both train and test datasets
    categorical_columns = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval', 'MonthPeriod']
    numeric_columns_test = ['Store', 'DayOfWeek', 'DaysTo_A_Holiday', 'SchoolHoliday',
                            'DaysAfter_A_Holiday', 'DaysTo_B_Holiday', 'DaysAfter_B_Holiday',
                            'DaysTo_C_Holiday', 'DaysAfter_C_Holiday', 'Open', 'Promo', 'Promo2',
                            'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day', 'WeekOfYear',
                            'IsWeekend', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                            'CompetitionOpenSinceYear']


    # 1. One-hot encode categorical columns and scale numeric columns for training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns_test),  # Apply to both train and test (numeric columns excluding Sales for test)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # handle_unknown='ignore' avoids unseen categories issues
        ]
    )

    test_data_preprocessed = preprocessor.fit_transform(test_data)

    encoded_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)

    test_columns = numeric_columns_test + list(encoded_cat_columns)


    test_data_preprocessed_df = pd.DataFrame(test_data_preprocessed, columns=test_columns)
    return test_data_preprocessed_df

def get_preprocessed_train_data(train_data):
    # Columns for both train and test datasets
    categorical_columns = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval', 'MonthPeriod']
    numeric_columns_train = ['Sales', 'Store', 'DayOfWeek', 'DaysTo_A_Holiday', 'SchoolHoliday',
                            'DaysAfter_A_Holiday', 'DaysTo_B_Holiday', 'DaysAfter_B_Holiday',
                            'DaysTo_C_Holiday', 'DaysAfter_C_Holiday', 'Open', 'Promo', 'Promo2',
                            'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day', 'WeekOfYear',
                            'IsWeekend', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                            'CompetitionOpenSinceYear']

    # numeric_columns_test = [col for col in numeric_columns_train if col != 'Sales']  # Exclude 'Sales'

    # 1. One-hot encode categorical columns and scale numeric columns for training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns_train),  # Apply to both train and test (numeric columns excluding Sales for test)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # handle_unknown='ignore' avoids unseen categories issues
        ]
    )

    # 2. Fit the preprocessor on the training data
    train_data_preprocessed = preprocessor.fit_transform(train_data)


    # 4. Recreate DataFrames for both train and test data
    # Get column names for the one-hot encoded categorical columns
    encoded_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)

    # Combine numeric and one-hot encoded column names
    train_columns = numeric_columns_train + list(encoded_cat_columns)


    # Create new DataFrames with the transformed data and appropriate column names
    train_data_preprocessed_df = pd.DataFrame(train_data_preprocessed, columns=train_columns)
    return train_data_preprocessed_df