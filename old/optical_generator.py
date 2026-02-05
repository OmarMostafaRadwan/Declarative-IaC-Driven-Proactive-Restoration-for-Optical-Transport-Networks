import random
import math
import csv
import numpy as np

# --- CONFIGURATION ---
OUTPUT_FILE = "optical_data.csv"
START_SNR = 25.0
DT = 1.0  # Time step (seconds)

# --- PHYSICS PARAMETERS (Ornstein-Uhlenbeck Process) ---
# dS(t) = theta * (mu - S(t)) * dt + sigma * dW(t)
# This models thermal stability: The signal "wants" to stay at MU (25.0),
# but thermal noise (SIGMA) pushes it away.
THETA = 0.1   # Speed of mean reversion
MU = 25.0     # The long-term mean SNR (Ideal state)
SIGMA = 0.2   # Volatility (Thermal noise)

def generate_soft_failure_data(duration_seconds):
    """
    Simulates: Laser Aging / Connector Oxidation (Soft Failure)
    Physics: Stochastic Drift-Diffusion with deterministic decay trend.
    """
    print(f"--- Generating SOFT FAILURE data for {duration_seconds} seconds ---")
    current_snr = START_SNR
    data_points = []
    
    # Failure starts at 30% mark
    failure_start = int(duration_seconds * 0.3)
    
    for t in range(duration_seconds):
        # 1. Stochastic Component (The SDE)
        # Random thermal noise (Wiener Process dW)
        dW = np.random.normal(0, math.sqrt(DT))
        # Mean Reversion (The drift back to stability)
        dx = THETA * (MU - current_snr) * DT + SIGMA * dW
        current_snr += dx
        
        # 2. Deterministic Component (The Failure)
        # We model aging as a linear drift superimposed on the stochastic noise
        if t > failure_start:
            # Aging Rate: 0.05 dB per second (Fast degradation for demo)
            aging_loss = 0.05 * (t - failure_start) / 10.0
            current_snr -= aging_loss

        # Store: [Time, SNR, Label]
        # Label 1 means "Pre-FEC Error Threshold" (typically < 18dB)
        label = 1 if current_snr < 18.0 else 0
        data_points.append([t, round(current_snr, 4), label])

    return data_points

def generate_normal_data(duration_seconds):
    print(f"--- Generating NORMAL data for {duration_seconds} seconds ---")
    current_snr = START_SNR
    data_points = []
    
    for t in range(duration_seconds):
        # Only SDE (No Failure Drift)
        dW = np.random.normal(0, math.sqrt(DT))
        dx = THETA * (MU - current_snr) * DT + SIGMA * dW
        current_snr += dx
        
        data_points.append([t, round(current_snr, 4), 0])
        
    return data_points

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Seconds", "SNR_dB", "Label"])
        writer.writerows(data)
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Generate Training Data
    normal_data = generate_normal_data(1000)
    save_to_csv(normal_data, "training_normal.csv")
    
    failure_data = generate_soft_failure_data(1000)
    save_to_csv(failure_data, "training_failure.csv")
    
    print("Data Generation Complete: Stochastic Model Applied.")