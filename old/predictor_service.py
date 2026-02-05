import time
import requests
import pandas as pd
import joblib
import numpy as np
import warnings
import os
import subprocess
import shap
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
MODEL_FILE = "optical_ai_model_v2.pkl"
WINDOW_SIZE = 10 
CRITICAL_THRESHOLD_SECONDS = 60.0 
PERSISTENCE_LIMIT = 3
persistence_counter = 0

print("--- Loading AI Model & XAI Explainer ---")
try:
    model = joblib.load(MODEL_FILE)
    # TreeExplainer is optimized for Random Forest
    explainer = shap.TreeExplainer(model)
    print("Model & Explainer loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find {MODEL_FILE}. Run train_ai_v2.py first!")
    exit()

def get_realtime_data():
    """
    Fetches the raw SNR metric from Prometheus (Time Series Database).
    """
    query_params = {'query': 'optical_snr_db[30s]'}
    try:
        response = requests.get(PROMETHEUS_URL, params=query_params)
        data = response.json()
        if data['status'] == 'success' and len(data['data']['result']) > 0:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', 'SNR_dB'])
            df['SNR_dB'] = df['SNR_dB'].astype(float)
            return df
        else:
            return None
    except Exception as e:
        return None

def prepare_features(df):
    """
    Transforms raw time-series into the Lag-based feature set the AI expects.
    Input: [t, SNR] -> Output: [SNR, SNR_Lag_1, ... SNR_Lag_10]
    """
    df = df.sort_values('timestamp')
    for i in range(1, WINDOW_SIZE + 1):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)
    
    last_row = df.tail(1).copy()
    
    # Ensure we have a full window of data (no NaNs)
    if last_row.isnull().values.any():
        return None
        
    feature_cols = [col for col in last_row.columns if 'SNR' in col]
    return last_row[feature_cols]

# --- Q1 UPGRADE: PHYSICS-BASED ENERGY MODEL ---
def calculate_power_consumption(snr):
    """
    Calculates dynamic power consumption of the Optical Transponder DSP.
    
    Physics Justification:
    Power scales with the number of Soft-Decision LDPC iterations required 
    to correct bit errors. This relationship is non-linear (The 'Turbo Cliff').
    """
    P_IDLE = 35.0  # Watts (Laser + Cooling baseline)
    P_MAX_DSP = 50.0 # Watts (Full Load DSP)
    
    # Thresholds for Modulation Switching
    SNR_IDEAL = 22.0    # Clean signal, minimal correction needed
    SNR_CRITICAL = 16.0 # Near failure, max correction effort
    
    if snr >= SNR_IDEAL:
        return P_IDLE
    elif snr <= SNR_CRITICAL:
        return P_MAX_DSP
    else:
        # Normalize the SNR drop to a 0.0-1.0 scale
        load_factor = (SNR_IDEAL - snr) / (SNR_IDEAL - SNR_CRITICAL)
        
        # Iteration count (and thus power) scales exponentially near the limit
        # We use a power of 1.5 to model the non-linear DSP load
        dynamic_power = (P_MAX_DSP - P_IDLE) * (load_factor ** 1.5)
        return P_IDLE + dynamic_power

# --- MAIN CONTROL LOOP ---
if __name__ == "__main__":
    print(f"--- Zero-Touch Restoration Agent (Green-Net Mode) ---")
    
    # Ensure Network is in Default State
    subprocess.run(["terraform.exe", "apply", "-var=active_path=primary", "-auto-approve"], stdout=subprocess.DEVNULL)
    
    energy_log = []

    while True:
        raw_df = get_realtime_data()
        
        # We need enough history to build the Lag features
        if raw_df is not None and len(raw_df) >= WINDOW_SIZE + 2:
            features = prepare_features(raw_df)
            
            if features is not None:
                current_snr = features['SNR_dB'].values[0]
                
                # 1. AI PREDICTION
                prediction = model.predict(features)[0]
                
                # 2. ENERGY CALCULATION (Sustainability Metric)
                current_power = calculate_power_consumption(current_snr)
                energy_log.append(current_power)
                
                # 3. EXPLAINABILITY (XAI)
                shap_values = explainer.shap_values(features)
                
                # 4. DECISION LOGIC WITH COOL-DOWN HYSTERESIS
                status = "SAFE"
                color = "\033[92m" # Green Text
                
                if prediction < CRITICAL_THRESHOLD_SECONDS:
                    status = "CRITICAL"
                    color = "\033[91m" # Red Text
                    persistence_counter += 1
                else:
                    # COOL-DOWN: Do not reset to 0 immediately. 
                    # Slowly decrease confidence counter to avoid "Flapping".
                    persistence_counter = max(0, persistence_counter - 1)
                
                print(f"{color}SNR: {current_snr:.2f}dB | Power: {current_power:.1f}W | TTF: {prediction:.1f}s | Alarm: {persistence_counter}/{PERSISTENCE_LIMIT}\033[0m")
                
                # 5. ORCHESTRATION TRIGGER
                if persistence_counter >= PERSISTENCE_LIMIT:
                    print(f"\n{color}>>> FAILURE TRAJECTORY CONFIRMED. INITIATING MIGRATION... <<<")
                    
                    # A. Generate Evidence (Artifacts for the Paper)
                    # Trust Artifact: Why did it trigger?
                    plt.figure(figsize=(10, 4))
                    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
                    plt.title(f"XAI: Trigger Explanation (SNR={current_snr:.2f}dB)")
                    plt.tight_layout()
                    plt.savefig("xai_explanation_graph.png")
                    
                    # Green Artifact: How much power was spiking?
                    plt.figure(figsize=(8, 4))
                    plt.plot(energy_log, color='orange', label='FEC Power (Watts)')
                    plt.axhline(y=35, color='green', linestyle='--', label='Baseline Power')
                    plt.title("Sustainability Analysis: Pre-empting Energy Spike")
                    plt.xlabel("Time (Samples)")
                    plt.ylabel("Power (Watts)")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("energy_efficiency_graph.png")
                    
                    print(f">>> Evidence Saved: XAI and Energy Graphs generated.")

                    # B. Execute Infrastructure-as-Code
                    print(f">>> EXECUTING TERRAFORM MIGRATION... <<<")
                    cmd = ["terraform.exe", "apply", "-var=active_path=backup", "-auto-approve"]
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"{color}>>> SUCCESS: TRAFFIC MIGRATED TO BACKUP PATH <<<\033[0m")
                        break # Exit loop after successful handling (or reset)
                    except Exception as e:
                        print(f"Terraform Failed: {e}")
        
        else:
            print("Buffering telemetry data...", end='\r')

        time.sleep(1)