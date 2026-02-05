import time
import random
import os
import math
import numpy as np
from prometheus_client import start_http_server, Gauge

# --- CONFIGURATION ---
EXPORTER_PORT = 8000
SNR_METRIC = Gauge('optical_snr_db', 'Current Signal-to-Noise Ratio')
TRIGGER_FILE = "break_link.txt" 

# --- PHYSICS CONSTANTS (MATCHING YOUR GENERATOR) ---
START_SNR = 25.0
THETA = 0.1   # Speed of mean reversion
MU = 25.0     # Ideal mean
SIGMA = 0.2   # Thermal volatility
DT = 1.0      # Time step

def run_simulation():
    start_http_server(EXPORTER_PORT)
    print(f"--- Optical Digital Twin (Ornstein-Uhlenbeck) Running on Port {EXPORTER_PORT} ---")
    print(f"To break the link, create: '{TRIGGER_FILE}'")
    
    current_snr = START_SNR
    time_in_failure = 0

    while True:
        # 1. CHECK FAILURE TRIGGER
        is_broken = os.path.exists(TRIGGER_FILE)

        # 2. CALCULATE PHYSICS (SDE)
        # Stochastic Component (Thermal Noise)
        dW = np.random.normal(0, math.sqrt(DT))
        # Mean Reversion (Drift to Stability)
        dx = THETA * (MU - current_snr) * DT + SIGMA * dW
        current_snr += dx

        if is_broken:
            # FAILURE PHYSICS (Deterministic Drift)
            time_in_failure += 1
            # Matches the generator's aging rate
            aging_loss = 0.05 * (time_in_failure) / 10.0
            current_snr -= aging_loss
            print(f"CRITICAL: Physical Degradation... SNR: {current_snr:.2f} dB")
        else:
            # RECOVERY PHYSICS
            time_in_failure = 0
            if current_snr < 24.0: 
                current_snr += 0.5 # Active Repair
            
            # Clamp for realism
            if current_snr > 26.0: current_snr = 26.0
            
            print(f"NORMAL: System Stable. SNR: {current_snr:.2f} dB")

        # 3. PUSH TO TELEMETRY
        SNR_METRIC.set(current_snr)
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()