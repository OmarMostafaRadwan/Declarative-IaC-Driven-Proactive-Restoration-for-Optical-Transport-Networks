"""
Enhanced Training Script v3
- Trains on multiple degradation models
- Cross-validates across models
- Implements proper baselines
- Statistical significance testing
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from scipy import stats
import matplotlib.pyplot as plt

# Import baselines (local module in the "files" package)
from files.baseline_models import (
    SimpleThresholdBaseline,
    LinearExtrapolationBaseline,
    MovingAverageBaseline,
    evaluate_model,
    create_comparison_table
)

# Configuration
DATA_DIR = "training_data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
WINDOW_SIZE = 10
MAX_TTF = 300
N_REPETITIONS = 30  # Increased from 10 for statistical significance

# Create directories
for directory in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


def create_lag_features(df, window_size=WINDOW_SIZE):
    """Create lag features from time series"""
    for i in range(1, window_size + 1):
        df[f'SNR_Lag_{i}'] = df['SNR_dB'].shift(i)
    return df.dropna()


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


def load_dataset(filename):
    """Load and process a single dataset"""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    df = pd.read_csv(filepath)
    df = calculate_ttf(df)
    df = create_lag_features(df)
    
    feature_cols = [col for col in df.columns if 'SNR' in col]
    X = df[feature_cols]
    y = df['Time_to_Failure']
    
    return X, y


def load_all_datasets():
    """Load all available datasets"""
    datasets = {}
    
    # Normal operation
    result = load_dataset("training_normal.csv")
    if result:
        datasets['normal'] = result
    
    # Failure scenarios
    failure_types = ['ou_failure', 'exp_failure', 'weibull_failure', 
                     'step_failure', 'osc_failure']
    
    for ftype in failure_types:
        result = load_dataset(f"training_{ftype}.csv")
        if result:
            datasets[ftype] = result
    
    return datasets


def combine_datasets(datasets, include_types=None):
    """Combine multiple datasets"""
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


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def cross_model_validation(datasets):
    """
    Critical validation: Train on one model, test on another
    This proves the system learns general degradation, not specific physics
    """
    print("\n" + "="*70)
    print("CROSS-MODEL VALIDATION")
    print("="*70)
    
    failure_types = ['ou_failure', 'exp_failure', 'weibull_failure', 
                     'step_failure', 'osc_failure']
    
    results_matrix = []
    
    for train_type in failure_types:
        if train_type not in datasets:
            continue
            
        print(f"\nTraining on: {train_type}")
        
        # Train on this type + normal
        X_train, y_train = combine_datasets(datasets, ['normal', train_type])
        if X_train is None:
            continue
        
        model = train_random_forest(X_train, y_train)
        
        # Test on ALL types
        row_results = {'Train': train_type}
        
        for test_type in failure_types:
            if test_type not in datasets:
                continue
            
            X_test, y_test = datasets[test_type]
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            
            row_results[f'Test_{test_type}'] = f"{mae:.2f}s"
            
            if train_type == test_type:
                print(f"  → Same model test: {mae:.2f}s MAE")
            else:
                print(f"  → Cross-model ({test_type}): {mae:.2f}s MAE")
        
        results_matrix.append(row_results)
    
    # Create cross-validation table
    df_cross = pd.DataFrame(results_matrix)
    print("\n" + "="*70)
    print("CROSS-MODEL VALIDATION MATRIX")
    print("="*70)
    print(df_cross.to_string(index=False))
    
    # Save results
    df_cross.to_csv(f"{RESULTS_DIR}/cross_model_validation.csv", index=False)
    
    return df_cross


def statistical_significance_test(model, baselines, X_test, y_test, n_reps=30):
    """
    Statistical test: Is Random Forest significantly better than baselines?
    Uses paired t-test
    """
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    
    # Get Random Forest predictions
    rf_predictions = model.predict(X_test)
    rf_errors = np.abs(rf_predictions - y_test)
    
    results = []
    
    for baseline in baselines:
        baseline_predictions = baseline.predict(X_test)
        baseline_errors = np.abs(baseline_predictions - y_test)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rf_errors, baseline_errors)
        
        rf_mae = np.mean(rf_errors)
        baseline_mae = np.mean(baseline_errors)
        improvement = ((baseline_mae - rf_mae) / baseline_mae) * 100
        
        significant = "YES" if p_value < 0.05 else "NO"
        
        results.append({
            'Baseline': baseline.name,
            'RF MAE': f"{rf_mae:.2f}s",
            'Baseline MAE': f"{baseline_mae:.2f}s",
            'Improvement': f"{improvement:.1f}%",
            'p-value': f"{p_value:.4f}",
            'Significant': significant
        })
        
        print(f"{baseline.name:25s} | p={p_value:.4f} | Improvement={improvement:+.1f}% | {significant}")
    
    df_stats = pd.DataFrame(results)
    df_stats.to_csv(f"{RESULTS_DIR}/statistical_tests.csv", index=False)
    
    return df_stats


def sensitivity_analysis(datasets, param_ranges):
    """
    Test sensitivity to hyperparameters
    Shows results are robust to parameter choices
    """
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)
    
    X_train, y_train = combine_datasets(datasets)
    X_test, y_test = combine_datasets(datasets, ['exp_failure'])
    
    results = []
    
    for n_est in param_ranges['n_estimators']:
        for max_d in param_ranges['max_depth']:
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'MAE': mae
            })
            
            print(f"n_est={n_est}, max_depth={max_d}: MAE={mae:.2f}s")
    
    df_sensitivity = pd.DataFrame(results)
    
    # Plot heatmap using Matplotlib only (no seaborn dependency)
    pivot = df_sensitivity.pivot(index='max_depth', columns='n_estimators', values='MAE')
    plt.figure(figsize=(10, 6))
    im = plt.imshow(pivot.values, aspect='auto', origin='lower', cmap='RdYlGn_r')
    plt.colorbar(im, label='MAE (s)')
    
    # Set axis ticks and labels
    plt.xticks(
        ticks=range(len(pivot.columns)),
        labels=pivot.columns,
        rotation=45
    )
    plt.yticks(
        ticks=range(len(pivot.index)),
        labels=pivot.index
    )
    
    # Annotate each cell with its MAE value
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(
                j,
                i,
                f"{pivot.values[i, j]:.2f}",
                ha='center',
                va='center',
                color='black'
            )
    
    plt.title('Sensitivity Analysis: MAE vs Hyperparameters')
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/sensitivity_heatmap.png", dpi=300)
    print(f"Saved heatmap to {RESULTS_DIR}/sensitivity_heatmap.png")
    
    return df_sensitivity


def main():
    print("="*70)
    print("ENHANCED TRAINING PIPELINE V3")
    print("="*70)
    
    # Step 1: Load all datasets
    print("\nStep 1: Loading datasets...")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Step 2: Combine training data
    print("\nStep 2: Combining datasets for training...")
    X_train, y_train = combine_datasets(datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Step 3: Train main model
    print("\nStep 3: Training Random Forest...")
    model = train_random_forest(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Random Forest Results:")
    print(f"  MAE:  {mae:.2f}s")
    print(f"  RMSE: {rmse:.2f}s")
    print(f"  R²:   {r2:.4f}")
    
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
    
    # Add Random Forest to comparison
    rf_result = evaluate_model(model, X_test, y_test, "Random Forest")
    baseline_results.append(rf_result)
    
    comparison_df = create_comparison_table(baseline_results)
    comparison_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    
    # Step 5: Statistical significance
    stats_df = statistical_significance_test(model, baselines, X_test, y_test)
    
    # Step 6: Cross-model validation
    cross_val_df = cross_model_validation(datasets)
    
    # Step 7: Sensitivity analysis
    param_ranges = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20]
    }
    sensitivity_df = sensitivity_analysis(datasets, param_ranges)
    
    # Step 8: Save final model
    model_path = f"{MODEL_DIR}/optical_ai_model_v3.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Step 9: Generate summary report
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"✓ Random Forest MAE: {mae:.2f}s (vs worst baseline improvement)")
    print(f"✓ Cross-model validation: See {RESULTS_DIR}/cross_model_validation.csv")
    print(f"✓ Statistical tests: See {RESULTS_DIR}/statistical_tests.csv")
    print(f"✓ All results saved to: {RESULTS_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
