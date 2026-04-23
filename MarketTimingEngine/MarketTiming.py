import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. SIMULATED TRAINING DATA (Historical)
# ==========================================
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

# ==========================================
# 2. MODEL TRAINING
# ==========================================
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







# Save the model for offline edge deployment (optional)
# model.save_model("market_timing_v1.json") 

# ==========================================
# 3. THE INFERENCE FUNCTION (For your Webhook)
# ==========================================
def get_dynamic_directive(trend_percentage, volatility, sma_7, sma_30, season_code, weather_code, commodity_name, days_held, model):
    """
    Calculates the market directive using smoothed features, penalized by spoilage risk.
    """
    # 1. Define the physical constraints
    shelf_life_db = {
        "Yellow Corn": 180,
        "Cabbage": 14,
        "Tomatoes": 7
    }
    
    max_days = shelf_life_db.get(commodity_name, 7)
    
    # 2. Calculate the Spoilage Risk (0.0 = Fresh, 1.0 = Rotten)
    spoilage_risk = (days_held / max_days) ** 2 
    
    # 3. Construct the expanded State DataFrame
    # The XGBoost model must be fed the exact columns it was trained on.
    current_state = pd.DataFrame({
        'trend_7d': [trend_percentage],
        'volatility_14d': [volatility],
        'sma_7d': [sma_7],
        'sma_30d': [sma_30],
        'seasonality': [season_code],
        'weather_shock': [weather_code]
    })
    
    # Predict probability of price going up (Class 1)
    raw_hold_prob = model.predict_proba(current_state)[0][1]
    
    # 4. Apply the Perishability Penalty
    adjusted_hold_prob = raw_hold_prob * (1 - spoilage_risk)
    
    print(f"DEBUG: {commodity_name} | Raw AI: {raw_hold_prob:.0%} | Spoilage Risk: {spoilage_risk:.0%} | Adjusted: {adjusted_hold_prob:.0%}")

    # 5. Dynamic Thresholding
    if spoilage_risk > 0.85:
        return f"SELL NOW: Your {commodity_name} is near spoilage. Do not wait for better prices."
    elif adjusted_hold_prob >= 0.70:
        return "WAIT: Prices are climbing and your crop is stable."
    elif adjusted_hold_prob >= 0.40:
        return "SELL SOON: Find a buyer. The risk of spoilage is outweighing potential price gains."
    else:
        return "SELL NOW: Prices are dropping fast. Take the best offer today."

def process_market_trends(csv_filepath, output_filepath="processed_trends.csv"):
    """
    Reads raw agricultural prices, handles missing calendar days, 
    and calculates trends, rolling volatility, and moving averages.
    """
    print(f"Loading raw data from {csv_filepath}...")
    
    df = pd.read_csv(csv_filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    processed_commodities = []

    for commodity_name, group in df.groupby('commodity'):
        
        group = group.set_index('date')
        group = group.sort_index()
        
        # 1. Handle missing days
        group = group.resample('D').ffill()
        
        # 2. Point-to-Point Trend
        group['trend_7d'] = group['price'].pct_change(periods=7)
        
        # 3. NOISE FILTERS (The New Features)
        # Volatility: Standard deviation over the last 14 days
        group['volatility_14d'] = group['price'].rolling(window=14).std()
        
        # Moving Averages: 7-day (fast) and 30-day (slow)
        group['sma_7d'] = group['price'].rolling(window=7).mean()
        group['sma_30d'] = group['price'].rolling(window=30).mean()
        
        group = group.reset_index()
        group['commodity'] = commodity_name
        processed_commodities.append(group)

    final_df = pd.concat(processed_commodities, ignore_index=True)
    
    # 4. The Data Hunger Drop
    # We must drop rows that contain NaN. Because sma_30d requires 30 days 
    # to calculate, the first 30 days of your dataset will be dropped.
    final_df = final_df.dropna(subset=['trend_7d', 'volatility_14d', 'sma_7d', 'sma_30d'])
    
    # Reorder columns for clean reading
    final_df = final_df[['date', 'commodity', 'price', 'trend_7d', 'volatility_14d', 'sma_7d', 'sma_30d']]
    
    final_df.to_csv(output_filepath, index=False)
    print(f"Processing complete! Saved to {output_filepath}")
    
    return final_df

def handle_farmer_sms(requested_commodity, current_season_code, current_weather_code, days_held, model, results_df):
    """
    Acts as the webhook bridge: pulls the latest processed features from the database
    and feeds them to the AI inference engine.
    """
    # 1. Filter the dataset for ONLY the commodity the farmer texted about
    commodity_data = results_df[results_df['commodity'] == requested_commodity]
    
    if commodity_data.empty:
        return f"Error: No recent data found for {requested_commodity}."
    
    # 2. Isolate the absolute last row of THAT specific commodity (the most recent day)
    latest_data = commodity_data.iloc[-1]
    
    # 3. Extract ALL required features
    current_trend = latest_data['trend_7d']
    current_volatility = latest_data['volatility_14d']
    current_sma_7 = latest_data['sma_7d']
    current_sma_30 = latest_data['sma_30d']
    current_date = latest_data['date'].strftime('%Y-%m-%d')
    
    print(f"DEBUG: SMS triggered for {current_date} | Trend: {current_trend:.2%} | Volatility: {current_volatility:.2f}")
    
    # 4. Feed everything into the XGBoost inference function
    directive = get_dynamic_directive(
        trend_percentage=current_trend, 
        volatility=current_volatility,
        sma_7=current_sma_7,
        sma_30=current_sma_30,
        season_code=current_season_code, 
        weather_code=current_weather_code,
        commodity_name=requested_commodity,
        days_held=days_held,
        model=model # Explicitly passing the trained model
    )
    
    return directive

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Assuming your raw data is named 'market_prices.csv'
    # It will output a new file named 'processed_trends.csv'
    try:
        results = process_market_trends("market_prices.csv")
        
        print("\nPreview of the processed data:")
        print(results.tail(10)) # Show the last 10 rows

        # ==========================================
        # 4. SIMULATING A REAL-WORLD SMS TRIGGER
        # ==========================================
        print("\n--- Testing the Inference Engine ---")

        final_sms_text = handle_farmer_sms(
            requested_commodity="Yellow Corn", 
            current_season_code=1, 
            current_weather_code=0, 
            days_held=17, # Testing a crop near its 180-day limit
            model=model, 
            results_df=results
        )
        print(f"\nFinal Output to SMS: {final_sms_text}")

    except FileNotFoundError:
        print("Error: Could not find 'market_prices.csv'. Please ensure the file is in the same directory.")

