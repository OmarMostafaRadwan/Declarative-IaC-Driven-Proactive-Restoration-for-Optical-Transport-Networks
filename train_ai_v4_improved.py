"""
train_ai_v4_improved.py
ADVANCED Training Script
- Adds Derivative Features (Velocity, Acceleration) to fix Weibull/Step performance
- Uses older sklearn syntax for compatibility
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from scipy import stats

# Import baselines (assumes baseline_models.py is in the same folder or 'files' package)
# If this fails, ensure baseline_models.py is in the directory
try:
    from files.baseline_models import (
        SimpleThresholdBaseline,
        LinearExtrapolationBaseline,
        MovingAverageBaseline,
        evaluate_model,
        create_comparison_table
    )
except ImportError:
    # Fallback if running directly without package structure
    from files.baseline_models import *

# Configuration
DATA_DIR = "training_data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
WINDOW_SIZE = 10
MAX_TTF = 300
N_REPETITIONS = 30

# Create directories
for directory in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


def calculate_ttf(df, threshold=15.0):
    """Calculate time-to-failure for each timestep"""
    failure_points = df[df['SNR_dB'] < threshold]
    
    if len(failure_points) > 0:
        crash_time = failure_points['Time_Seconds'].min()
    else:
        crash_time = df['Time_Seconds'].max() + MAX_TTF
    
    df['Time_to_Failure'] = crash_time - df['Time_Seconds']
    df['Time_to_Failure'] = df['Time_to_Failure'].clip(upper=MAX_TTF)
    df.loc[df['Time_to_Failure'] < 0, 'Time_to_Failure'] = 0
    return df


def create_advanced_features(df, window_size=WINDOW_SIZE):
    """
    Create advanced features: Lags + Velocity + Acceleration + Volatility
    This helps the model distinguish Step (high velocity) from Weibull (high acceleration)
    """
    # 1. Standard Lags
    for i in range(1, window_size + 1):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)
    
    # Drop NaNs created by lags first to ensure valid calc for derivatives
    df = df.dropna().copy()

    # 2. Velocity (First Derivative): Current - Lag1
    df['Velocity'] = df['SNR_dB'] - df['SNR_Lag_1']
    
    # 3. Acceleration (Second Derivative): Velocity - Prev_Velocity
    # We reconstruct prev_velocity from lags
    prev_velocity = df['SNR_Lag_1'] - df['SNR_Lag_2']
    df['Acceleration'] = df['Velocity'] - prev_velocity
    
    # 4. Rolling Statistics (Volatile vs Smooth)
    # Use rolling window on the Lag_1 to avoid data leakage from "future" (current row is safe)
    # calculating std dev of last 5 lags
    lag_cols = [f'SNR_Lag_{i}' for i in range(1, 6)]
    df['Rolling_Std'] = df[lag_cols].std(axis=1)
    df['Rolling_Mean'] = df[lag_cols].mean(axis=1)
    
    return df.dropna()


def load_dataset_v4(filename):
    """Load and process a single dataset with advanced features"""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    df = pd.read_csv(filepath)
    df = calculate_ttf(df)
    df = create_advanced_features(df)  # USE NEW FEATURE FUNCTION
    
    feature_cols = [col for col in df.columns if 'SNR' in col or 'Velocity' in col or 'Acceleration' in col or 'Rolling' in col]
    X = df[feature_cols]
    y = df['Time_to_Failure']
    
    return X, y


def load_all_datasets_v4():
    """Load all available datasets"""
    datasets = {}
    
    # Normal operation
    result = load_dataset_v4("training_normal.csv")
    if result:
        datasets['normal'] = result
    
    # Failure scenarios
    failure_types = ['ou_failure', 'exp_failure', 'weibull_failure', 
                     'step_failure', 'osc_failure']
    
    for ftype in failure_types:
        result = load_dataset_v4(f"training_{ftype}.csv")
        if result:
            datasets[ftype] = result
    
    return datasets


def combine_datasets(datasets, include_types=None):
    if include_types is None:
        include_types = list(datasets.keys())
    
    X_list = []
    y_list = []
    
    for dtype in include_types:
        if dtype in datasets:
            X, y = datasets[dtype]
            X_list.append(X)
            y_list.append(y)
    
    if not X_list:
        return None, None
    
    X_combined = pd.concat(X_list, ignore_index=True)
    y_combined = pd.concat(y_list, ignore_index=True)
    
    return X_combined, y_combined


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=150,       # Increased slightly
        max_depth=12,           # Increased slightly to capture complex boundaries
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def main():
    print("="*70)
    print("IMPROVED TRAINING PIPELINE V4 (With Derivatives)")
    print("="*70)
    
    # Step 1: Load all datasets
    print("\nStep 1: Loading datasets with ADVANCED features...")
    datasets = load_all_datasets_v4()
    
    # Step 2: Combine training data
    print("\nStep 2: Combining datasets for training...")
    X_train, y_train = combine_datasets(datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Step 3: Train main model
    print("\nStep 3: Training Random Forest (v4)...")
    model = train_random_forest(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Random Forest V4 Results:")
    print(f"  MAE:  {mae:.2f}s  (Target: < 17s)")
    print(f"  RMSE: {rmse:.2f}s")
    print(f"  R²:   {r2:.4f}")
    
    # Compare with Step Function specifically (the hardest case)
    if 'step_failure' in datasets:
        X_step, y_step = datasets['step_failure']
        step_pred = model.predict(X_step)
        step_mae = mean_absolute_error(y_step, step_pred)
        print(f"  Step Failure MAE: {step_mae:.2f}s (Did features help?)")

    # Step 4: Train baselines
    print("\nStep 4: Training baseline models...")
    baselines = [
        SimpleThresholdBaseline(),
        LinearExtrapolationBaseline(window_size=5),
        MovingAverageBaseline(window_size=10)
    ]
    
    baseline_results = []
    for baseline in baselines:
        baseline.fit(X_train, y_train)
        result = evaluate_model(baseline, X_test, y_test)
        baseline_results.append(result)
    
    # Add RF
    rf_result = evaluate_model(model, X_test, y_test, "Random Forest v4")
    baseline_results.append(rf_result)
    
    comparison_df = create_comparison_table(baseline_results)
    comparison_df.to_csv(f"{RESULTS_DIR}/model_comparison_v4.csv", index=False)
    
    # Step 5: Save
    model_path = f"{MODEL_DIR}/optical_ai_model_v4.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()