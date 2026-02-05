import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --- CONFIGURATION ---
NORMAL_FILE = "training_normal.csv"
FAILURE_FILE = "training_failure.csv"
MODEL_FILENAME = "optical_ai_model_v2.pkl"
# Scientific Trick: Cap the prediction. 
# We don't care if it fails in 100 hours. We care if it fails in 5 minutes (300s).
MAX_TTF = 300 

def create_lag_features(df, window_size=5):
    """
    This gives the AI 'Memory'.
    Instead of just seeing [Current_SNR], it sees:
    [Current_SNR, SNR_1s_Ago, SNR_2s_Ago, ... SNR_5s_Ago]
    """
    for i in range(1, window_size + 1):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)
    
    # Drop the first few rows that have NaN (empty) values because of the shift
    df = df.dropna()
    return df

def load_and_process_data():
    print("--- Loading and Processing Data (Version 2) ---")
    
    # 1. Load Normal Data
    df_normal = pd.read_csv(NORMAL_FILE)
    df_normal['Time_to_Failure'] = MAX_TTF # Cap at 300s
    
    # 2. Load Failure Data
    df_failure = pd.read_csv(FAILURE_FILE)
    
    # Calculate TTF for failure
    failure_point = df_failure[df_failure['SNR_dB'] < 15.0]
    if len(failure_point) > 0:
        crash_time = failure_point['Time_Seconds'].min()
    else:
        crash_time = df_failure['Time_Seconds'].max()

    df_failure['Time_to_Failure'] = crash_time - df_failure['Time_Seconds']
    
    # Cap the TTF at 300s. 
    # If TTF is 500s, the AI just learns "It's safe (300)".
    # If TTF is 60s, the AI learns "It's urgent (60)".
    df_failure['Time_to_Failure'] = df_failure['Time_to_Failure'].clip(upper=MAX_TTF)
    df_failure.loc[df_failure['Time_to_Failure'] < 0, 'Time_to_Failure'] = 0

    # Combine
    df_final = pd.concat([df_normal, df_failure])
    
    # 3. ADD MEMORY (The Key Fix)
    # We create a window of the last 10 seconds
    df_final = create_lag_features(df_final, window_size=10)

    # Define Input (X) and Output (y)
    # X is now [SNR, Lag_1, Lag_2... Lag_10]
    feature_cols = [col for col in df_final.columns if 'SNR' in col]
    X = df_final[feature_cols]
    y = df_final['Time_to_Failure']
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_process_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("--- Training AI with Memory (Rolling Window) ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    
    print(f"Model V2 Trained! Average Error: +/- {error:.2f} seconds")
    
    joblib.dump(model, MODEL_FILENAME)