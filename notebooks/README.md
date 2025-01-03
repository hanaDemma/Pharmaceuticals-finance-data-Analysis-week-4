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

### Task 1.2 - Logging
The goal is to log all steps using the `logging` library in Python for traceability and reproducibility.

#### Logging Best Practices:
- Log the start and end of major processes.
- Include checkpoints for critical EDA steps (e.g., missing value checks, outlier detection).
- Log summary statistics and key findings at each stage.
- Ensure logs are written to a file for persistent traceability.
