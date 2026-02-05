"""
generate_figures.py
Creates all publication-quality figures for the paper
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Configuration
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def figure_1_scenario_detection():
    """
    Figure 1: Scenario Testing - Detection Reliability
    Shows SNR evolution with detection markers for each scenario
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    scenarios = [
        ("OU Drift", "ou_failure", 47.2),
        ("Exponential Decay", "exp_failure", 79.6),
        ("Weibull Acceleration", "weibull_failure", 34.4),
        ("Catastrophic Step", "step_failure", -3.0)
    ]
    
    for ax, (name, model, lead_time) in zip(axes.flat, scenarios):
        # Load data
        df = pd.read_csv(f'training_data/training_{model}.csv')
        
        # Plot SNR
        ax.plot(df['Time_Seconds'], df['SNR_dB'], 
                linewidth=1.5, alpha=0.8, label='SNR', color='blue')
        
        # Thresholds
        ax.axhline(15, color='red', linestyle='--', 
                   linewidth=2, label='Failure (15dB)', alpha=0.7)
        ax.axhline(18, color='orange', linestyle='--', 
                   linewidth=2, label='Warning (18dB)', alpha=0.7)
        
        # Detection marker
        if lead_time > 0:
            # Find first failure
            failure_time = df[df['SNR_dB'] < 15]['Time_Seconds'].min()
            detection_time = failure_time - lead_time
            
            ax.axvline(detection_time, color='green', linestyle='-', 
                       linewidth=2, alpha=0.5, label=f'Detection ({lead_time:.0f}s lead)')
            ax.axvline(failure_time, color='red', linestyle='-', 
                       linewidth=2, alpha=0.5, label='Actual Failure')
        
        ax.set_title(f'{name}\nLead Time: {lead_time:.1f}s')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('SNR (dB)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([10, 28])
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure1_scenario_detection.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 1 saved: Scenario Detection")


def figure_2_energy_savings():
    """
    Figure 2: Energy Savings vs Gamma Sensitivity
    """
    df = pd.read_csv('results/energy_validation_results.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart at gamma=1.5
    df_15 = df[df['Gamma'] == 1.5].sort_values('Avg_Joules_Saved', ascending=False)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax1.bar(range(len(df_15)), df_15['Avg_Joules_Saved'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_xticks(range(len(df_15)))
    ax1.set_xticklabels(df_15['Scenario'], rotation=15, ha='right')
    ax1.set_ylabel('Energy Saved (Joules)')
    ax1.set_title('Energy Savings per Event (γ=1.5)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_15['Avg_Joules_Saved'])):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, 
                 f'{val:.1f}J', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Sensitivity to gamma
    scenarios = df['Scenario'].unique()
    for scenario in scenarios:
        data = df[df['Scenario'] == scenario].sort_values('Gamma')
        ax2.plot(data['Gamma'], data['Avg_Joules_Saved'], 
                 marker='o', linewidth=2, label=scenario, markersize=8)
    
    ax2.set_xlabel('Gamma Exponent (Turbo Cliff Steepness)')
    ax2.set_ylabel('Energy Saved (Joules)')
    ax2.set_title('Sensitivity to FEC Decoder Model')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1.0, 1.3, 1.5, 1.8, 2.0])
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure2_energy_savings.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 2 saved: Energy Savings")


def figure_3_model_comparison():
    """
    Figure 3: Model Comparison - Prediction Error Distributions
    """
    # Recreate predictions for visualization
    # This requires loading models and test data - simplified version here
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAE Comparison Bar Chart
    df = pd.read_csv('results/model_comparison_v4.csv')
    
    models = df['Model'].values
    maes = df['MAE'].values
    
    colors = ['green' if 'Random' in m else 'gray' for m in models]
    bars = ax1.barh(models, maes, color=colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Mean Absolute Error (seconds)')
    ax1.set_title('Prediction Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, maes):
        ax1.text(val + 5, bar.get_y() + bar.get_height()/2, 
                 f'{val:.1f}s', va='center', fontweight='bold')
    
    # Plot 2: R² Comparison
    r2_vals = df['R²'].values
    
    colors = ['green' if 'Random' in m else 'red' if r2 < 0 else 'orange' 
              for m, r2 in zip(models, r2_vals)]
    bars = ax2.barh(models, r2_vals, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('R² Score (Variance Explained)')
    ax2.set_title('Model Quality (R²)')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.axvline(0.9, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax2.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, r2_vals):
        x_pos = val + 0.05 if val > 0 else val - 0.1
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{val:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure3_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 3 saved: Model Comparison")


def figure_4_cross_validation():
    """
    Figure 4: Cross-Model Validation Heatmap
    """
    df = pd.read_csv('results/cross_model_validation.csv')
    
    # Extract numeric values from strings
    train_models = df['Train'].values
    
    # Create matrix
    test_cols = [c for c in df.columns if 'Test_' in c]
    matrix = np.zeros((len(train_models), len(test_cols)))
    
    for i, row in df.iterrows():
        for j, col in enumerate(test_cols):
            val_str = row[col].replace('s', '')
            matrix[i, j] = float(val_str)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
    
    # Labels
    ax.set_xticks(range(len(test_cols)))
    ax.set_yticks(range(len(train_models)))
    ax.set_xticklabels([c.replace('Test_', '').replace('_', ' ').title() 
                        for c in test_cols], rotation=45, ha='right')
    ax.set_yticklabels([t.replace('_', ' ').title() for t in train_models])
    
    # Annotations
    for i in range(len(train_models)):
        for j in range(len(test_cols)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('Test Model')
    ax.set_ylabel('Train Model')
    ax.set_title('Cross-Model Validation Matrix (MAE in seconds)')
    
    plt.colorbar(im, ax=ax, label='MAE (seconds)')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure4_cross_validation.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 4 saved: Cross-Model Validation")


def figure_5_lead_time_distribution():
    """
    Figure 5: Lead Time Distribution Across Scenarios
    """
    df = pd.read_csv('results/scenario_results.csv')
    
    # Filter only successful detections
    df_success = df[df['Result'] == 'Success']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Box plot by scenario
    scenarios = df_success['Scenario'].str.extract(r'(.+?) \(Run')[0].unique()
    data_by_scenario = []
    labels = []
    
    for scenario in scenarios:
        mask = df_success['Scenario'].str.contains(scenario)
        lead_times = df_success[mask]['Lead_Time'].values
        if len(lead_times) > 0:
            data_by_scenario.append(lead_times)
            labels.append(scenario.replace(' ', '\n'))
    
    bp = ax1.boxplot(data_by_scenario, labels=labels, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax1.set_ylabel('Lead Time (seconds)')
    ax1.set_title('Lead Time Distribution by Scenario')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(60, color='green', linestyle='--', alpha=0.5, label='Target (60s)')
    ax1.legend()
    
    # Plot 2: Histogram of all lead times
    all_lead_times = df_success['Lead_Time'].values
    ax2.hist(all_lead_times, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(all_lead_times), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_lead_times):.1f}s')
    ax2.axvline(np.median(all_lead_times), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(all_lead_times):.1f}s')
    
    ax2.set_xlabel('Lead Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Lead Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure5_lead_time_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 5 saved: Lead Time Distribution")


def main():
    print("="*70)
    print("GENERATING ALL PAPER FIGURES")
    print("="*70)
    
    try:
        figure_1_scenario_detection()
    except Exception as e:
        print(f"⚠️ Figure 1 failed: {e}")
    
    try:
        figure_2_energy_savings()
    except Exception as e:
        print(f"⚠️ Figure 2 failed: {e}")
    
    try:
        figure_3_model_comparison()
    except Exception as e:
        print(f"⚠️ Figure 3 failed: {e}")
    
    try:
        figure_4_cross_validation()
    except Exception as e:
        print(f"⚠️ Figure 4 failed: {e}")
    
    try:
        figure_5_lead_time_distribution()
    except Exception as e:
        print(f"⚠️ Figure 5 failed: {e}")
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print(f"All figures saved to: {FIGURES_DIR}/")
    print("\nNext step: Insert figures into paper with proper captions")


if __name__ == "__main__":
    main()
