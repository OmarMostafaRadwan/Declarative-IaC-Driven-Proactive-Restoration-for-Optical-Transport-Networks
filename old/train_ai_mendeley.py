import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --- CONFIGURATION ---
# Replace this with the actual path to your downloaded text file
DATA_FILE = "Lightpath_756_label_4_QoT_dataset_test_300.txt"
MODEL_FILENAME = "optical_ai_model_mendeley.pkl"
WINDOW_SIZE = 10  # Look back 10 samples
MAX_TTF_CAP = 300 # Cap predictions at 300 seconds (5 minutes)

def parse_mendeley_data(filepath):
    """
    Parses the specific 'Optical Network Soft Failure' text format.
    Structure: Space-separated values, header on line 2 (index 1).
    """
    print(f"--- Parsing Mendeley Data: {filepath} ---")
    
    # 1. Read the file
    # We skip line 0 (Description) and use line 1 as header.
    # 'delim_whitespace=True' handles the spaces between columns.
    df = pd.read_csv(filepath, skiprows=1, delim_whitespace=True)
    
    # 2. Clean Column Names
    # Remove quotes from column names: '"Time stamp"' -> 'Time stamp'
    df.columns = df.columns.str.replace('"', '').str.strip()
    
    # 3. Select & Rename Relevant Columns
    # We map "OSNR (dB)" to our model's "SNR_dB" input
    # We map "Time stamp" to "Time_Seconds"
    df = df.rename(columns={
        'OSNR (dB)': 'SNR_dB',
        'Time stamp': 'Time_Seconds',
        'Failure type': 'Label'
    })
    
    # Ensure data is numeric
    df['SNR_dB'] = pd.to_numeric(df['SNR_dB'], errors='coerce')
    df['Time_Seconds'] = pd.to_numeric(df['Time_Seconds'], errors='coerce')
    
    return df

def calculate_ttf(df):
    """
    Logic: Find the moment the 'Label' changes from 0 (Normal) to anything else.
    That moment is the 'Crash'.
    """
    # 1. Find the first failure timestamp
    # Label 0 = Normal. Label 1,2,3 = Failure.
    failure_rows = df[df['Label'] != 0]
    
    if len(failure_rows) > 0:
        # The crash happens at the FIRST timestamp where label != 0
        crash_time = failure_rows['Time_Seconds'].min()
        print(f"FAILURE DETECTED at T={crash_time}s (First non-zero label)")
    else:
        # If no failure in file, assume it survives until the end
        crash_time = df['Time_Seconds'].max()
        print("NO FAILURE DETECTED in file. Assuming safe.")

    # 2. Calculate Time-to-Failure (TTF)
    # TTF = Crash_Time - Current_Time
    df['Time_to_Failure'] = crash_time - df['Time_Seconds']
    
    # 3. Clean up
    # If time is past the crash (negative TTF), set to 0
    df.loc[df['Time_to_Failure'] < 0, 'Time_to_Failure'] = 0
    
    # 4. Cap the TTF (Scientific Trick)
    # We don't want the AI to predict "10000 seconds left". 
    # We just want it to know if it's > 300s (Safe) or < 60s (Critical).
    df['Time_to_Failure'] = df['Time_to_Failure'].clip(upper=MAX_TTF_CAP)
    
    return df

def create_lag_features(df, window_size=5):
    """
    Adds memory to the dataset (Lag Features).
    """
    df = df.sort_values('Time_Seconds')
    for i in range(1, window_size + 1):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)
    
    return df.dropna()

if __name__ == "__main__":
    # --- STEP 1: LOAD & PARSE ---
    try:
        raw_df = parse_mendeley_data(DATA_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Make sure the file is in the same folder!")
        exit()
        
    # --- STEP 2: CALCULATE TARGET (TTF) ---
    processed_df = calculate_ttf(raw_df)
    
    # --- STEP 3: ADD FEATURES (LAGS) ---
    final_df = create_lag_features(processed_df, window_size=WINDOW_SIZE)
    
    # --- STEP 4: PREPARE TRAINING DATA ---
    # Inputs (X): SNR and its Lags
    feature_cols = [col for col in final_df.columns if 'SNR' in col]
    X = final_df[feature_cols]
    
    # Output (y): Time to Failure
    y = final_df['Time_to_Failure']
    
    # --- STEP 5: TRAIN ---
    print(f"--- Training on {len(X)} Real-World Samples ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- STEP 6: EVALUATE ---
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"SUCCESS: Real-Data Model Trained!")
    print(f"Mean Absolute Error: +/- {error:.2f} seconds")
    
    # --- STEP 7: SAVE ---
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")