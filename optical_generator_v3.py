"""
Enhanced Optical Digital Twin - Multi-Model Version
Implements multiple degradation physics models to avoid circular validation
"""
import random
import math
import csv
import numpy as np
from enum import Enum

class DegradationModel(Enum):
    """Different physical degradation models"""
    ORNSTEIN_UHLENBECK = "ou"  # Original model
    EXPONENTIAL_DECAY = "exp"   # Realistic amplifier aging
    WEIBULL_PROCESS = "weibull" # Component reliability model
    STEP_FUNCTION = "step"      # Catastrophic failure
    OSCILLATORY = "osc"         # Amplifier instability
    MULTI_PARAMETER = "multi"   # SNR + CD + PMD combined

# --- CONFIGURATION ---
OUTPUT_DIR = "training_data"
START_SNR = 25.0
DT = 1.0  # Time step (seconds)

# --- OU PARAMETERS (Original) ---
THETA = 0.1   # Speed of mean reversion
MU = 25.0     # Long-term mean SNR
SIGMA = 0.2   # Volatility (Thermal noise)

# --- EXPONENTIAL DECAY PARAMETERS ---
LAMBDA_EXP = 0.003  # Decay constant (realistic: weeks to months)
TIME_SCALE_FACTOR = 100  # Speed up for simulation (100x faster)

# --- WEIBULL PARAMETERS ---
WEIBULL_SHAPE = 2.5  # Shape parameter (k)
WEIBULL_SCALE = 500  # Scale parameter (λ) in seconds

# --- OSCILLATORY PARAMETERS ---
OSC_AMPLITUDE = 2.0  # Oscillation amplitude in dB
OSC_FREQUENCY = 0.05  # Oscillations per second
OSC_DAMPING = 0.001  # Damping factor


class OpticalDigitalTwin:
    """
    Multi-model digital twin for optical network simulation
    """
    
    def __init__(self, model_type: DegradationModel, duration_seconds: int):
        self.model_type = model_type
        self.duration = duration_seconds
        self.current_snr = START_SNR
        self.data_points = []
        
    def add_thermal_noise(self):
        """Common thermal noise component (Wiener process)"""
        dW = np.random.normal(0, math.sqrt(DT))
        return SIGMA * dW
    
    def ou_drift(self):
        """Ornstein-Uhlenbeck mean reversion"""
        return THETA * (MU - self.current_snr) * DT
    
    def generate_ou_failure(self, failure_start_ratio=0.3):
        """Original OU model with linear aging"""
        print(f"Generating OU model data ({self.duration}s)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            # Stochastic component
            drift = self.ou_drift()
            noise = self.add_thermal_noise()
            self.current_snr += drift + noise
            
            # Deterministic aging
            if t > failure_start:
                aging_loss = 0.05 * (t - failure_start) / 10.0
                self.current_snr -= aging_loss
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_exponential_failure(self, failure_start_ratio=0.3):
        """
        Exponential decay model (realistic amplifier/laser aging)
        Physics: P(t) = P0 * exp(-λt)
        More realistic than linear degradation
        """
        print(f"Generating EXPONENTIAL decay model ({self.duration}s)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            # Stochastic baseline
            drift = self.ou_drift()
            noise = self.add_thermal_noise()
            self.current_snr += drift + noise
            
            # Exponential degradation
            if t > failure_start:
                time_since_failure = (t - failure_start)
                # Accelerated exponential decay
                decay = START_SNR * (1 - np.exp(-LAMBDA_EXP * TIME_SCALE_FACTOR * time_since_failure / self.duration))
                self.current_snr = START_SNR - decay + noise
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_weibull_failure(self, failure_start_ratio=0.3):
        """
        Weibull process (component reliability model)
        Hazard rate: h(t) = (k/λ) * (t/λ)^(k-1)
        Used in reliability engineering for component failures
        """
        print(f"Generating WEIBULL process model ({self.duration}s)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            drift = self.ou_drift()
            noise = self.add_thermal_noise()
            self.current_snr += drift + noise
            
            if t > failure_start:
                time_since_failure = (t - failure_start)
                # Weibull cumulative hazard
                hazard = (time_since_failure / WEIBULL_SCALE) ** WEIBULL_SHAPE
                degradation = 10.0 * (1 - np.exp(-hazard))  # Scale to reach threshold
                self.current_snr = START_SNR - degradation + noise
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_step_failure(self):
        """
        Step function (catastrophic failure - fiber cut)
        Instant degradation with no warning
        Tests if model can handle sudden changes
        """
        print(f"Generating STEP function (catastrophic) ({self.duration}s)...")
        failure_time = int(self.duration * 0.5)  # Fail at midpoint
        
        for t in range(self.duration):
            if t < failure_time:
                # Normal operation with noise
                drift = self.ou_drift()
                noise = self.add_thermal_noise()
                self.current_snr += drift + noise
            else:
                # Sudden drop
                if t == failure_time:
                    self.current_snr = 12.0  # Instant drop below threshold
                # Small noise afterward
                self.current_snr += self.add_thermal_noise() * 0.1
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_oscillatory_failure(self, failure_start_ratio=0.3):
        """
        Oscillatory degradation (amplifier instability)
        Combines sinusoidal oscillation with damped decay
        Tests model's ability to handle periodic patterns
        """
        print(f"Generating OSCILLATORY degradation ({self.duration}s)...")
        failure_start = int(self.duration * failure_start_ratio)
        
        for t in range(self.duration):
            drift = self.ou_drift()
            noise = self.add_thermal_noise()
            self.current_snr += drift + noise
            
            if t > failure_start:
                time_since_failure = (t - failure_start)
                # Damped oscillation
                oscillation = OSC_AMPLITUDE * np.sin(2 * np.pi * OSC_FREQUENCY * time_since_failure)
                damping = np.exp(-OSC_DAMPING * time_since_failure)
                trend = -0.02 * time_since_failure  # Slow linear decay
                
                self.current_snr += (oscillation * damping + trend)
            
            label = 1 if self.current_snr < 18.0 else 0
            self.data_points.append([t, round(self.current_snr, 4), label])
        
        return self.data_points
    
    def generate_normal_operation(self):
        """Normal operation with only stochastic noise (no failure)"""
        print(f"Generating NORMAL operation ({self.duration}s)...")
        
        for t in range(self.duration):
            drift = self.ou_drift()
            noise = self.add_thermal_noise()
            self.current_snr += drift + noise
            
            # Clamp to realistic range
            self.current_snr = max(20.0, min(28.0, self.current_snr))
            
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
    print(f"Saved {filename} ({len(data)} samples)")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("ENHANCED OPTICAL DIGITAL TWIN - MULTI-MODEL GENERATOR")
    print("="*60)
    
    # Generate normal operation data
    twin_normal = OpticalDigitalTwin(DegradationModel.ORNSTEIN_UHLENBECK, 1000)
    normal_data = twin_normal.generate_normal_operation()
    save_to_csv(normal_data, f"{OUTPUT_DIR}/training_normal.csv")
    
    # Generate MULTIPLE failure scenarios
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
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Location: {OUTPUT_DIR}/")
    print("Next step: Train models on these diverse scenarios")
