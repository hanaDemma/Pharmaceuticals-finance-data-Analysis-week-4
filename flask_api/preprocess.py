import pandas as pd

def preprocess_input(data, scaler):
    # Convert input JSON to DataFrame
    df = pd.DataFrame(data)
    
    # Extract features (ensure these match the training features)
    features = df[["Store", " DayOfWeek", "DaysTo_A_Holiday", "SchoolHoliday", "DaysAfter_A_Holiday", "DaysTo_B_Holiday", "DaysAfter_B_Holiday", "DaysTo_C_Holiday", "DaysAfter_C_Holiday", "Open", "Promo", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "Year", "Month", "Day", "WeekOfYear", "IsWeekend", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "StoreType_a", "StoreType_b", "StoreType_c", 
                   "StoreType_d", "Assortment_a", "Assortment_b", "Assortment_c", "StateHoliday_0", "StateHoliday_a", "StateHoliday_b", "StateHoliday_c", "PromoInterval_Feb_May_Aug_Nov", "PromoInterval_Jan_Apr_Jul_Oct", "PromoInterval_Mar_Jun_Sept_Dec", "MonthPeriod_beginning", "MonthPeriod_end", "MonthPeriod_middle"]]
   

    # Scale the features
    features_scaled = scaler.transform(features)
    
    return features_scaled