"""
scenario_tester.py
Tests the model against specific "What-If" scenarios
"""
import pandas as pd
import numpy as np
import joblib
import os
from optical_generator_v3_FIXED import OpticalDigitalTwin, DegradationModel

# Load your best model
model = joblib.load("models/optical_ai_model_v4.pkl")

def test_specific_scenario(name, model_type, duration=1000):
    print(f"Testing Scenario: {name}...")
    # Generate data
    twin = OpticalDigitalTwin(model_type, duration)
    data = twin.generate()

    # Process features (MANUALLY replicate v4 feature engineering)
    df = pd.DataFrame(data, columns=["Time_Seconds", "SNR_dB", "Label"])

    # Feature Engineering (Must match train_ai_v4 exactly!)
    for i in range(1, 11):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)

    df = df.dropna().copy()
    df['Velocity'] = df['SNR_dB'] - df['SNR_Lag_1']
    prev_velocity = df['SNR_Lag_1'] - df['SNR_Lag_2']
    df['Acceleration'] = df['Velocity'] - prev_velocity
    lag_cols = [f'SNR_Lag_{i}' for i in range(1, 6)]
    df['Rolling_Std'] = df[lag_cols].std(axis=1)
    df['Rolling_Mean'] = df[lag_cols].mean(axis=1)
    df = df.dropna()

    # Predict
    features = [c for c in df.columns if 'SNR' in c or 'Velocity' in c or 'Acceleration' in c or 'Rolling' in c]
    X = df[features]
    preds = model.predict(X)

    # Calculate Lead Time (Time between first prediction < 60s and actual failure)
    # Simple heuristic for this test
    avg_pred = np.mean(preds)
    print(f"  -> Average Predicted TTF: {avg_pred:.2f}s")
    return preds

if __name__ == "__main__":
    test_specific_scenario("Fast Decay (Exp)", DegradationModel.EXPONENTIAL_DECAY)
    test_specific_scenario("Slow Drift (OU)", DegradationModel.ORNSTEIN_UHLENBECK)
    print("\nScenario tests complete.")