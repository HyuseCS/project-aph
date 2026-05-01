import xgboost as xgb
import pandas as pd
from datetime import datetime, timedelta

def get_dynamic_directive(trend_percentage, volatility, sma_7, sma_30, season_code, weather_code, commodity_name, days_held, model, current_price, current_date_str):
    """
    Calculates the market directive using smoothed features, penalized by spoilage risk,
    and forecasts future prices to find the peak date.
    """
    # 1. Define the physical constraints
    shelf_life_db = {
        "Rice": 365,
        "Kamote": 30,
        "Cabbage": 14,
        "Tomato": 7
    }
    
    max_days = shelf_life_db.get(commodity_name, 7)
    days_left = max(0, max_days - days_held)
    
    # 2. Calculate the Spoilage Risk (0.0 = Fresh, 1.0 = Rotten)
    spoilage_risk = (days_held / max_days) ** 2 
    
    # 3. Forecast prices for the next 14 days
    future_prices = []
    for day_offset in range(1, 15):
        current_state = pd.DataFrame({
            'trend_7d': [trend_percentage],
            'volatility_14d': [volatility],
            'sma_7d': [sma_7],
            'sma_30d': [sma_30],
            'seasonality': [season_code],
            'weather_shock': [weather_code],
            'day_offset': [day_offset]
        })
        predicted_price = model.predict(current_state)[0]
        future_prices.append(predicted_price)
    
    peak_price = max(future_prices)
    peak_day_offset = future_prices.index(peak_price) + 1
    
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
    peak_date = current_date + timedelta(days=peak_day_offset)
    peak_date_str = peak_date.strftime('%B %d')
    
    # Simple logic: Is it worth waiting?
    expected_gain = peak_price - current_price
    
    print(f"DEBUG: {commodity_name} | Peak: P{peak_price:.2f} in {peak_day_offset} days | Spoilage Risk: {spoilage_risk:.0%}")

    # 5. Dynamic Thresholding
    price_info = f"{commodity_name} is currently P{current_price:.2f}/kg."
    
    if spoilage_risk > 0.85 or days_left <= 2:
        return f"SELL NOW: {price_info} Your crop will spoil in about {days_left} days. Do not wait."
    elif expected_gain > 0.5 and peak_day_offset > days_left-2:
        return f"SELL SOON: {price_info} The price might peak at P{peak_price:.2f} on {peak_date_str}, but your crop will likely spoil before then."
    elif expected_gain > 0.5:
        return f"WAIT: {price_info} Our models predict the price will peak around {peak_date_str} at ~P{peak_price:.2f}/kg. Your crop is stable."
    else:
        return f"SELL NOW: {price_info} Prices are not expected to rise significantly. Take the best offer today."


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

def handle_farmer_sms(requested_commodity, current_season, current_weather, days_held, model, results_df):
    """
    Acts as the webhook bridge: pulls the latest processed features from the database
    and feeds them to the AI inference engine.
    """
    season_map = {"dry": 1, "wet": 2}
    weather_map = {"sunny": 0, "normal": 0, "drought": 1, "flood": 2}
    
    season_code = season_map.get(current_season.lower().strip(), 1)
    weather_code = weather_map.get(current_weather.lower().strip(), 0)

    # 1. Filter the dataset for ONLY the commodity the farmer texted about (case-insensitive)
    commodity_data = results_df[results_df['commodity'].str.lower() == requested_commodity.lower().strip()]
    
    if commodity_data.empty:
        return f"Error: No recent data found for {requested_commodity}."
    
    # 2. Isolate the absolute last row of THAT specific commodity (the most recent day)
    latest_data = commodity_data.iloc[-1]
    
    # Extract the properly cased commodity name from the dataset
    actual_commodity_name = latest_data['commodity']
    
    # 3. Extract ALL required features
    current_trend = latest_data['trend_7d']
    current_volatility = latest_data['volatility_14d']
    current_sma_7 = latest_data['sma_7d']
    current_sma_30 = latest_data['sma_30d']
    current_price = latest_data['price']
    current_date = latest_data['date'].strftime('%Y-%m-%d')
    
    print(f"DEBUG: SMS triggered for {current_date} | Trend: {current_trend:.2%} | Volatility: {current_volatility:.2f}")
    
    # 4. Feed everything into the XGBoost inference function
    directive = get_dynamic_directive(
        trend_percentage=current_trend, 
        volatility=current_volatility,
        sma_7=current_sma_7,
        sma_30=current_sma_30,
        season_code=season_code, 
        weather_code=weather_code,
        commodity_name=actual_commodity_name,
        days_held=days_held,
        model=model, # Explicitly passing the trained model
        current_price=current_price,
        current_date_str=current_date
    )
    
    return directive


if __name__ == "__main__":
    try:
        results = process_market_trends("market_prices.csv")
        
        print("\nPreview of the processed data:")
        print(results.tail(10)) # Show the last 10 rows

        print("\n--- Testing the Inference Engine ---")

        model = xgb.XGBRegressor()
        model.load_model("market_timing_v2.json") # Load the trained model

        final_sms_text = handle_farmer_sms(
            requested_commodity="kamote", 
            current_season="wet", 
            current_weather="flood", 
            days_held=4, 
            model=model, 
            results_df=results
        )
        print(f"\nFinal Output to SMS: {final_sms_text}")

    except FileNotFoundError:
        print("Error: Could not find 'market_prices.csv'. Please ensure the file is in the same directory.")

