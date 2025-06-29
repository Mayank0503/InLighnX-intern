import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet  

df = pd.read_csv("air_pollution_data.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Replace -200 with NaN (if applicable)
df.replace(-200, np.nan, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

# Keep relevant columns
df = df[['date', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['date']))
scaled_df = pd.DataFrame(scaled_features, columns=df.columns[1:])
scaled_df['date'] = df['date']

# 5. Prophet Forecasting Model (Target = AQI)
prophet_df = df[['date', 'aqi']].rename(columns={'date': 'ds', 'aqi': 'y'})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
plt.title("AQI Forecast - Prophet")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

merged = forecast[['ds', 'yhat']].merge(prophet_df, on='ds', how='left')
merged.dropna(inplace=True)

mae = mean_absolute_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
r2 = r2_score(merged['y'], merged['yhat'])

print(f"\nModel Evaluation Metrics:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.2f}")

plt.figure(figsize=(12,6))
plt.plot(merged['ds'], merged['y'], label='Actual AQI')
plt.plot(merged['ds'], merged['yhat'], label='Predicted AQI', linestyle='--')
plt.title("Actual vs Predicted Air Quality Index (AQI)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
