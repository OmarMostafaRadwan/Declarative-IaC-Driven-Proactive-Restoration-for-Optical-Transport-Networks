"""
FIXED Optical Digital Twin - Multi-Model Version
Ensures all failure models actually reach threshold
"""
import random
import math
import csv
import numpy as np
from enum import Enum

class DegradationModel(Enum):
    ORNSTEIN_UHLENBECK = "ou"
    EXPONENTIAL_DECAY = "exp"
    WEIBULL_PROCESS = "weibull"
    STEP_FUNCTION = "step"
    OSCILLATORY = "osc"

# Configuration
OUTPUT_DIR = "training_data"
START_SNR = 25.0
DT = 1.0

# OU Parameters
THETA = 0.1
MU = 25.0
SIGMA = 0.2

# CRITICAL: Ensure all models reach threshold by end
TARGET_FINAL_SNR = 13.0  # Below 15dB threshold
FAILURE_THRESHOLD = 15.0


class OpticalDigitalTwin:
    def __init__(self, model_type: DegradationModel, duration_seconds: int):
        self.model_type = model_type
        self.duration = duration_seconds
        self.current_snr = START_SNR
        self.data_points = []
        
    def add_thermal_noise(self):
        """Common thermal noise component"""
        dW = np.random.normal(0, math.sqrt(DT))
        return SIGMA * dW
    
    def ou_drift(self):
        """Ornstein-Uhlenbeck mean reversion"""
        return THETA * (MU - self.current_snr) * DT
    
    def generate_ou_failure(self, failure_start_ratio=0.3):
        """
        FIXED OU model - ensures it reaches threshold
        Linear degradation calibrated to reach 13dB by end
        """
        print(f"Generating OU model (FIXED)...")
        failure_start = int(self.duration * failure_start_ratio)
        failure_duration = self.duration - failure_start
        
        # Calculate required degradation rate
        total_degradation_needed = START_SNR - TARGET_FINAL_SNR  # 25 - 13 = 12dB
        degradation_per_step = total_degradation_needed / failure_duration
        
        for t in range(self.duration):
            # Stochastic baseline (small noise)
            noise = self.add_thermal_noise() * 0.3  # Reduced noise
            
            if t < failure_start:
                # Normal operation - stay near START_SNR
                self.current_snr = START_SNR + noise
            else:
                # Linear degradation
                time_in_failure = t - failure_start
                self.current_snr = START_SNR - (degradation_per_step * time_in_failure) + noise
            
            # Clamp to reasonable range
            self.current_snr = max(10.0, min(28.0, self.current_snr))
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_exponential_failure(self, failure_start_ratio=0.3):
        """
        FIXED Exponential decay - ensures threshold is reached
        Uses exponential curve that goes from 25dB to 13dB
        """
        print(f"Generating EXPONENTIAL decay (FIXED)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            noise = self.add_thermal_noise() * 0.3
            
            if t < failure_start:
                self.current_snr = START_SNR + noise
            else:
                time_in_failure = (t - failure_start)
                failure_duration = self.duration - failure_start
                
                # Exponential decay formula: SNR(t) = START + (TARGET - START) * (1 - e^(-kt))
                # where k is chosen so that at t=failure_duration, we reach TARGET
                k = 3.0 / failure_duration  # Tuned to reach ~95% of target by end
                
                decay_factor = 1 - np.exp(-k * time_in_failure)
                self.current_snr = START_SNR + (TARGET_FINAL_SNR - START_SNR) * decay_factor + noise
            
            self.current_snr = max(10.0, min(28.0, self.current_snr))
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_weibull_failure(self, failure_start_ratio=0.3):
        """
        FIXED Weibull process - guaranteed to reach threshold
        Uses Weibull CDF shape
        """
        print(f"Generating WEIBULL process (FIXED)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            noise = self.add_thermal_noise() * 0.3
            
            if t < failure_start:
                self.current_snr = START_SNR + noise
            else:
                time_in_failure = (t - failure_start)
                failure_duration = self.duration - failure_start
                
                # Weibull CDF: F(t) = 1 - exp(-(t/λ)^k)
                # We want F(failure_duration) ≈ 1, so choose λ appropriately
                shape = 2.5  # k parameter
                scale = failure_duration / 2.0  # λ parameter
                
                if time_in_failure == 0:
                    weibull_factor = 0
                else:
                    weibull_factor = 1 - np.exp(-((time_in_failure / scale) ** shape))
                
                self.current_snr = START_SNR + (TARGET_FINAL_SNR - START_SNR) * weibull_factor + noise
            
            self.current_snr = max(10.0, min(28.0, self.current_snr))
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_step_failure(self):
        """
        FIXED Step function - catastrophic drop
        """
        print(f"Generating STEP function (FIXED)...")
        failure_time = int(self.duration * 0.5)
        
        for t in range(self.duration):
            noise = self.add_thermal_noise() * 0.3
            
            if t < failure_time:
                self.current_snr = START_SNR + noise
            else:
                # Sudden drop to below threshold
                self.current_snr = 12.0 + noise  # Well below 15dB
            
            self.current_snr = max(10.0, min(28.0, self.current_snr))
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_oscillatory_failure(self, failure_start_ratio=0.3):
        """
        FIXED Oscillatory degradation - damped sine wave with downward trend
        """
        print(f"Generating OSCILLATORY degradation (FIXED)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            noise = self.add_thermal_noise() * 0.3
            
            if t < failure_start:
                self.current_snr = START_SNR + noise
            else:
                time_in_failure = (t - failure_start)
                failure_duration = self.duration - failure_start
                
                # Overall downward trend
                linear_component = START_SNR + (TARGET_FINAL_SNR - START_SNR) * (time_in_failure / failure_duration)
                
                # Add damped oscillation
                amplitude = 2.0
                frequency = 0.05
                damping = np.exp(-2.0 * time_in_failure / failure_duration)
                oscillation = amplitude * damping * np.sin(2 * np.pi * frequency * time_in_failure)
                
                self.current_snr = linear_component + oscillation + noise
            
            self.current_snr = max(10.0, min(28.0, self.current_snr))
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_normal_operation(self):
        """Normal operation - stays around 25dB"""
        print(f"Generating NORMAL operation...")
        
        for t in range(self.duration):
            noise = self.add_thermal_noise()
            self.current_snr = START_SNR + noise
            
            # Clamp to realistic range
            self.current_snr = max(22.0, min(28.0, self.current_snr))
            
            self.data_points.append([t, round(self.current_snr, 4), 0])
        
        return self.data_points
    
    def generate(self):
        """Generate data based on selected model"""
        if self.model_type == DegradationModel.ORNSTEIN_UHLENBECK:
            return self.generate_ou_failure()
        elif self.model_type == DegradationModel.EXPONENTIAL_DECAY:
            return self.generate_exponential_failure()
        elif self.model_type == DegradationModel.WEIBULL_PROCESS:
            return self.generate_weibull_failure()
        elif self.model_type == DegradationModel.STEP_FUNCTION:
            return self.generate_step_failure()
        elif self.model_type == DegradationModel.OSCILLATORY:
            return self.generate_oscillatory_failure()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def save_to_csv(data, filename):
    """Save generated data to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Seconds", "SNR_dB", "Label"])
        writer.writerows(data)
    
    # Quick validation
    import pandas as pd
    df = pd.read_csv(filename)
    below_threshold = (df['SNR_dB'] < FAILURE_THRESHOLD).sum()
    print(f"   ✓ Saved {filename} ({len(data)} samples, {below_threshold} below threshold)")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("FIXED OPTICAL DIGITAL TWIN - MULTI-MODEL GENERATOR")
    print("="*70)
    print(f"Target: All failure models will reach SNR < {FAILURE_THRESHOLD}dB")
    print("="*70)
    
    # Generate normal operation
    twin_normal = OpticalDigitalTwin(DegradationModel.ORNSTEIN_UHLENBECK, 1000)
    normal_data = twin_normal.generate_normal_operation()
    save_to_csv(normal_data, f"{OUTPUT_DIR}/training_normal.csv")
    
    # Generate failure scenarios
    failure_models = [
        (DegradationModel.ORNSTEIN_UHLENBECK, "ou_failure"),
        (DegradationModel.EXPONENTIAL_DECAY, "exp_failure"),
        (DegradationModel.WEIBULL_PROCESS, "weibull_failure"),
        (DegradationModel.STEP_FUNCTION, "step_failure"),
        (DegradationModel.OSCILLATORY, "osc_failure"),
    ]
    
    for model_type, name in failure_models:
        twin = OpticalDigitalTwin(model_type, 1000)
        data = twin.generate()
        save_to_csv(data, f"{OUTPUT_DIR}/training_{name}.csv")
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"Location: {OUTPUT_DIR}/")
    print("\nNext step: Run diagnose_data.py to verify datasets are correct")
