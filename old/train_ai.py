import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --- CONFIGURATION ---
NORMAL_FILE = "training_normal.csv"
FAILURE_FILE = "training_failure.csv"
MODEL_FILENAME = "optical_ai_model.pkl"

def load_and_label_data():
    print("--- Loading Data ---")
    
    # 1. Load Normal Data (Healthy)
    # We teach the AI that normal data has a "Time to Failure" of 9999 (Safe)
    df_normal = pd.read_csv(NORMAL_FILE)
    df_normal['Time_to_Failure'] = 9999 
    
    # 2. Load Failure Data (The Crash)
    df_failure = pd.read_csv(FAILURE_FILE)
    
    # Logic: We need to calculate how many seconds are left until the SNR drops below 15.0
    # Find the exact second where the link 'died' (SNR < 15.0)
    failure_point = df_failure[df_failure['SNR_dB'] < 15.0]
    
    if len(failure_point) > 0:
        crash_time = failure_point['Time_Seconds'].min()
    else:
        crash_time = df_failure['Time_Seconds'].max() # Use end of file if it never fully crashed

    # Calculate Time to Failure (TTF) for each row
    # Example: If crash is at 250s, and current time is 200s, TTF = 50s.
    df_failure['Time_to_Failure'] = crash_time - df_failure['Time_Seconds']
    
    # If the simulation continued AFTER the crash, set TTF to 0
    df_failure.loc[df_failure['Time_to_Failure'] < 0, 'Time_to_Failure'] = 0

    # Combine both datasets
    df_final = pd.concat([df_normal, df_failure])
    
    # We only train on "SNR_dB" to predict "Time_to_Failure"
    # In a real paper, we might use "Last 10 SNR values" (History), 
    # but for this prototype, simple SNR is enough to start.
    X = df_final[['SNR_dB']] 
    y = df_final['Time_to_Failure']
    
    return X, y

# --- MAIN TRAINING LOOP ---
if __name__ == "__main__":
    # 1. Prepare Data
    X, y = load_and_label_data()
    
    # 2. Split into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize the Brain (Random Forest)
    print("--- Training the AI Model (Random Forest) ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Model Trained! Average Error: +/- {error:.2f} seconds")
    
    # 5. Save the Brain
    joblib.dump(model, MODEL_FILENAME)
    print(f"Success. Model saved to {MODEL_FILENAME}")