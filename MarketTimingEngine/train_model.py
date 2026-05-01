import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_dummy_data(num_samples=2000):
    np.random.seed(42)
    
    # The Original Features
    trend_7d = np.random.uniform(-0.10, 0.10, num_samples)
    seasonality = np.random.choice([0, 1, 2], num_samples)
    weather_shock = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    
    # The New Noise Filter Features (Simulated)
    volatility_14d = np.random.uniform(0.5, 3.0, num_samples)
    sma_7d = np.random.uniform(15.0, 50.0, num_samples)
    sma_30d = sma_7d + np.random.uniform(-5.0, 5.0, num_samples)
    
    # Forward day offset (1 to 14 days into the future)
    day_offset = np.random.randint(1, 15, num_samples)
    
    # Create a simulated future price with non-linear cycles
    # 1. Base price from SMA
    # 2. Linear trend component
    # 3. Sine wave component (a 10-day cycle)
    # 4. Seasonal and weather shocks
    
    cycle_effect = 2.0 * np.sin(2 * np.pi * (day_offset / 10.0)) # 10-day cycle
    future_price = sma_7d * (1 + trend_7d * (day_offset / 14.0)) 
    future_price += cycle_effect
    future_price += seasonality * 1.5 + weather_shock * 5.0
    future_price += np.random.normal(0, volatility_14d, num_samples)
    future_price = np.maximum(future_price, 1.0) # Ensure no negative prices
    
    return pd.DataFrame({
        'trend_7d': trend_7d,
        'volatility_14d': volatility_14d,
        'sma_7d': sma_7d,
        'sma_30d': sma_30d,
        'seasonality': seasonality,
        'weather_shock': weather_shock,
        'day_offset': day_offset,
        'target_price': future_price
    })

if __name__ == "__main__":
    print("Generating data and training the XGBoost Regressor model...")
    df = generate_dummy_data()

    X = df[['trend_7d', 'volatility_14d', 'sma_7d', 'sma_30d', 'seasonality', 'weather_shock', 'day_offset']]
    y = df['target_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    model.save_model("market_timing_v2.json")
    print("Model training complete and saved as 'market_timing_v2.json'.")