import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    
    # The Original Features
    trend_7d = np.random.uniform(-0.10, 0.10, num_samples)
    seasonality = np.random.choice([0, 1, 2], num_samples)
    weather_shock = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    
    # The New Noise Filter Features (Simulated)
    volatility_14d = np.random.uniform(0.5, 3.0, num_samples)
    sma_7d = np.random.uniform(15.0, 25.0, num_samples)
    sma_30d = np.random.uniform(15.0, 25.0, num_samples)
    
    # Target Logic Update: If volatility is extremely high, penalize the hold probability
    hold_probability = (trend_7d * 5) + (seasonality * 0.2) + (weather_shock * 0.4) - (volatility_14d * 0.1)
    target = (hold_probability > 0.3).astype(int)
    
    return pd.DataFrame({
        'trend_7d': trend_7d,
        'volatility_14d': volatility_14d,
        'sma_7d': sma_7d,
        'sma_30d': sma_30d,
        'seasonality': seasonality,
        'weather_shock': weather_shock,
        'target': target
    })

if __name__ == "__main__":
    print("Generating data and training the XGBoost model...")
    df = generate_dummy_data()

    # Ensure all 6 features are passed to the model
    X = df[['trend_7d', 'volatility_14d', 'sma_7d', 'sma_30d', 'seasonality', 'weather_shock']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=50, 
        max_depth=3, 
        learning_rate=0.1, 
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)

    model.save_model("market_timing_v1.json")
    print("Model training complete and saved as 'market_timing_v1.json'.")