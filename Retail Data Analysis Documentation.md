# Retail Data Analysis Documentation

## 1. Introduction

This document provides comprehensive documentation for the `IDA_Assignment` GitHub repository, which contains a Python-based solution for analyzing an Online Retail Dataset. The analysis covers data cleaning, various performance analyses, visualization techniques, and statistical tests, aligning with the requirements outlined in the provided `Quiz # 04 (SE-417 IDA).docx` file.

The repository includes the following key files:
- `Complete_Retail_Analysis.py`: The main Python script performing all data analysis tasks.
- `Retail_Data_Analysis.ipynb`: A Jupyter Notebook version of the analysis (not directly used in this documentation but noted).
- `dataset.csv` and `dataset for quiz 4.csv`: The datasets used for the analysis.
- `README.md`: General information about the repository.

## 2. Setup and Execution

To run the analysis, ensure you have Python 3.x installed along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `sklearn`

You can install these using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

Navigate to the `IDA_Assignment` directory and execute the `Complete_Retail_Analysis.py` script:
```bash
python Complete_Retail_Analysis.py
```

The script will output the analysis results to the console and save various visualization plots as PNG files in the same directory.

## 3. Detailed Analysis (Based on `Quiz # 04 (SE-417 IDA).docx` Questions)

### Q1: Data Cleaning and Preprocessing

The `Complete_Retail_Analysis.py` script performs a series of essential data cleaning and preprocessing steps to prepare the `dataset for quiz 4.csv` for analysis. These steps ensure data quality and suitability for subsequent analytical tasks.

**Steps Performed:**
1.  **Loading Data**: The script loads the `dataset for quiz 4.csv` into a pandas DataFrame.
2.  **Missing Value Analysis**: It checks for and reports any missing values in the dataset. The provided script indicates that no missing values are found.
3.  **Duplicate Removal**: Duplicate rows are identified and removed from the dataset to prevent biased analysis.
4.  **Data Type Conversion and Feature Engineering**: The `Order Date` column is converted to datetime objects. New features are extracted from the `Order Date`, including `Year`, `Month`, `Month_Name`, `Quarter`, `Day_of_Week`, and `Is_Weekend`. Additionally, `Profit_Margin_Pct`, `Revenue_Per_Unit`, `Profit_Per_Unit`, and `Order_Value_Category` are engineered to provide deeper insights.
5.  **Outlier Detection**: Outliers in the `Amount` column are detected using the Interquartile Range (IQR) method. These outliers are noted but kept for analysis, as they might represent significant transactions.
6.  **Saving Cleaned Data**: The preprocessed dataset is saved as `dataset.csv`.

### Q2: Perform the following analysis on the given dataset.

The script conducts a comprehensive analysis across four key areas:

#### Sales Performance Analysis

This section examines various aspects of sales to understand performance trends and identify key drivers.

**Key Analyses:**
-   **Monthly Trends**: Aggregated monthly sales, profit, and quantity are calculated to observe temporal patterns.
-   **Category Performance**: Revenue, profit, and profit margin percentage are analyzed by product category to identify top-performing and underperforming categories.
-   **Payment Mode Analysis**: Total revenue, average transaction value, and transaction count are analyzed by payment mode to understand customer payment preferences and their impact on sales.
-   **Top Sub-Categories**: The top 10 sub-categories by revenue are identified.

#### Customer Retention and Churn Analysis

This analysis focuses on understanding customer behavior, loyalty, and potential churn.

**Key Analyses:**
-   **Customer Purchase Frequency**: The number of orders, total spent, first order date, and last order date are calculated for each customer to understand purchasing habits.
-   **Customer Lifetime Value (CLV)**: Customers are segmented into Low, Medium, and High CLV categories based on their total spending.
-   **Churn Analysis**: Customers who have not placed an order in the last 180 days are identified as churned, and the churn rate is calculated.
-   **Repeat Purchase Rate**: The percentage of customers who have made more than one purchase is calculated.

#### Sales Forecasting

This section attempts to predict future sales based on historical data.

**Key Analyses:**
-   **Time Series Data Preparation**: Monthly total sales (`Amount`) are prepared as time series data.
-   **Time Series Decomposition**: If sufficient data is available (at least 12 data points), the time series is decomposed into trend, seasonal, and residual components to understand underlying patterns.
-   **Simple Moving Average Forecast**: A 3-month moving average is used to forecast sales for the next month.
-   **Forecast Accuracy Metrics**: Mean Absolute Error (MAE) is calculated to evaluate the accuracy of the forecast.

#### Geographical Sales Analysis

This analysis explores sales performance across different geographical regions.

**Key Analyses:**
-   **State-wise Performance**: Total revenue, average order value, total profit, and order count are analyzed for each state to identify top-performing regions.
-   **City-level Analysis**: The top 10 cities by revenue are identified.
-   **State x Category Revenue Matrix**: A matrix showing revenue generated by each category in different states is created to identify regional product preferences.

### Q3: Apply different visualization techniques and explain how it can help you gain insights from the dataset.

The script generates several visualizations to provide intuitive insights into the data. These visualizations are saved as PNG files.

**Visualizations Generated:**
1.  **Monthly Revenue Trends (Line Chart)**: Displays the trend of total revenue over months, helping to identify seasonality and overall growth or decline.
2.  **Total Revenue by Category (Bar Chart)**: Compares the total revenue generated by different product categories, highlighting the most profitable ones.
3.  **Total Profit by Payment Mode (Bar Chart)**: Illustrates the profit contribution of each payment method, aiding in optimizing payment strategies.
4.  **Customer Purchase Frequency (Histogram)**: Shows the distribution of how many times customers purchase, indicating customer loyalty and engagement.
5.  **Sales by State (Bar Chart)**: Visualizes total sales across different states, pinpointing geographical strengths and weaknesses.
6.  **Sales by Day of Week (Bar Chart)**: Displays sales patterns across days of the week, useful for staffing and promotional scheduling.
7.  **Correlation Heatmap**: A heatmap showing the correlation between numerical features, helping to understand relationships between variables.

Each visualization provides a quick and effective way to grasp complex data patterns, identify anomalies, and communicate findings to stakeholders without delving into raw numbers.

### Q4: How can you perform different statistical tests to increase the sale in context of the given dataset.

The script applies various statistical tests to validate hypotheses and derive actionable recommendations for increasing sales.

**Statistical Tests Performed and Their Implications:**
1.  **ANOVA (Analysis of Variance) - Category vs. Amount**: Tests if there's a significant difference in average sales (`Amount`) across different product `Categories`. A significant p-value (e.g., < 0.05) suggests that certain categories perform significantly better or worse than others, guiding product focus and marketing efforts.
2.  **ANOVA - Payment Mode vs. Amount**: Examines if the average transaction `Amount` differs significantly based on the `Payment Mode`. A significant result indicates that promoting certain payment methods could lead to higher average order values.
3.  **Independent Samples t-test - Weekend vs. Weekday Sales**: Compares the average sales `Amount` between weekends and weekdays. A significant difference can inform optimal timing for promotions and resource allocation.
4.  **Chi-Squared Test - Category vs. Payment Mode**: Assesses if there's a significant association between `Category` and `Payment Mode`. This can reveal if certain payment methods are preferred for specific product types, allowing for targeted payment options.
5.  **Pearson Correlation - Amount vs. Profit**: Measures the linear relationship between `Amount` and `Profit`. A strong positive correlation implies that higher sales amounts generally lead to higher profits, reinforcing strategies to increase order value.
6.  **Linear Regression - Sales Prediction Model**: A linear regression model is built to predict `Amount` based on `Category`, `Payment Mode`, `State`, `Month`, and `Quantity`. This model helps understand the impact of these features on sales and can be used for forecasting and identifying key drivers. The R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are used to evaluate model performance, and feature coefficients indicate the importance of each factor.

These statistical tests provide a data-driven foundation for strategic recommendations, moving beyond mere observation to statistically validated insights that can directly influence sales growth strategies.

## 4. Strategic Recommendations for Sales Improvement

Based on the comprehensive analysis and statistical tests, the script provides several actionable recommendations:

1.  **Product Strategy**: Focus on high-performing categories (identified via ANOVA) and consider bundling low-margin products with high-margin ones.
2.  **Pricing & Profitability**: Leverage the strong correlation between order value and profit by implementing minimum order value discounts to increase transaction size. Expand on categories with the highest profit margins.
3.  **Payment Optimization**: Promote payment modes with the highest average transaction values and offer incentives for high-value payment methods.
4.  **Geographic Expansion**: Analyze successful states and replicate their strategies. Allocate marketing budgets proportional to regional performance.
5.  **Temporal Strategies**: Adjust promotional schedules based on whether weekend or weekday sales are higher and leverage identified seasonal patterns.
6.  **Customer Retention**: Implement win-back campaigns for churned customers, create loyalty programs, and focus personalized offers on high-CLV customers.
7.  **Forecasting & Inventory**: Utilize time series models for demand forecasting, prepare for seasonal variations, and optimize stock levels based on predicted demand.

This documentation provides a thorough overview of the `IDA_Assignment` repository's solution, detailing its analytical approach and the insights derived, all in line with the requirements of the `Quiz # 04 (SE-417 IDA).docx` document.
