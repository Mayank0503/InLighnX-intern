import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

df = pd.read_csv("bitcoin.csv")  # Change path as needed
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)

# Feature Engineering
df['SMA_14'] = df['Close'].rolling(window=14).mean()
df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
df['BB_upper'] = df['SMA_14'] + 2 * df['Close'].rolling(window=14).std()
df['BB_lower'] = df['SMA_14'] - 2 * df['Close'].rolling(window=14).std()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# Normalize Features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_14', 'EMA_14', 'BB_upper', 'BB_lower', 'RSI']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Train-Test Split
X = df_scaled.drop(['Close'], axis=1)
y = df_scaled['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Evaluation Function
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))
    return y_pred

# Model Training and Evaluation

print("\n🔹 Linear Regression:")
lr_model = LinearRegression()
evaluate_model(lr_model)

print("\n🔹 Random Forest:")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf_model)

print("\n🔹 Support Vector Machine (SVM):")
svm_model = SVR()
evaluate_model(svm_model)

print("\n🔹 XGBoost:")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
evaluate_model(xgb_model)
