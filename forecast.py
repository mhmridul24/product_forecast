import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# =========================
# LOAD CLEAN DATASET
# =========================

df = pd.read_csv("clean_usage_data.csv")

print("\n===== CLEAN DATASET =====")
print(df.head())

# =========================
# SELECT PRODUCT
# =========================

product_name = "Chicken"

product_df = df[df['Product'] == product_name]

print(f"\n===== {product_name} DATA =====")
print(product_df.head())

# =========================
# PREPARE DATA FOR PROPHET
# =========================

prophet_df = product_df[['Date', 'Usage']]

# Prophet requires:
# ds = date
# y = target value

prophet_df.columns = ['ds', 'y']

# Convert ds to datetime

prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

print("\n===== PROPHET DATA =====")
print(prophet_df.head())

# =========================
# CREATE MODEL
# =========================

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

# =========================
# TRAIN MODEL
# =========================

model.fit(prophet_df)

print("\nModel training completed!")

# =========================
# CREATE FUTURE DATES
# =========================

future = model.make_future_dataframe(periods=7)

# =========================
# PREDICT FUTURE
# =========================

forecast = model.predict(future)

# =========================
# SHOW FORECAST
# =========================

forecast_result = forecast[['ds', 'yhat']].tail(7)

print("\n===== NEXT 7 DAYS FORECAST =====")
print(forecast_result)

# =========================
# SAVE FORECAST
# =========================

forecast_result.to_csv("forecast_output.csv", index=False)

print("\nForecast saved as forecast_output.csv")

# =========================
# PLOT FORECAST
# =========================

fig = model.plot(forecast)

plt.title(f"{product_name} Usage Forecast")

plt.xlabel("Date")

plt.ylabel("Predicted Usage")

plt.show()