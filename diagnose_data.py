"""
Data Diagnostics Tool
Identifies issues with generated datasets
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def diagnose_dataset(filename):
    """Comprehensive dataset diagnostics"""
    filepath = os.path.join("training_data", filename)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTICS: {filename}")
    print(f"{'='*70}")
    
    # Basic stats
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # SNR analysis
    print(f"\n📈 SNR Statistics:")
    print(f"   Min:    {df['SNR_dB'].min():.2f} dB")
    print(f"   Max:    {df['SNR_dB'].max():.2f} dB")
    print(f"   Mean:   {df['SNR_dB'].mean():.2f} dB")
    print(f"   Std:    {df['SNR_dB'].std():.2f} dB")
    
    # Check if SNR ever crosses threshold
    threshold_crossings = (df['SNR_dB'] < 15.0).sum()
    print(f"\n⚠️  Samples below 15dB threshold: {threshold_crossings} ({threshold_crossings/len(df)*100:.1f}%)")
    
    if threshold_crossings == 0:
        print(f"   🚨 CRITICAL: SNR never crosses failure threshold!")
        print(f"   This means ALL TTF values will be capped at 300s")
        print(f"   Model has nothing to learn!")
    
    # Label distribution
    if 'Label' in df.columns:
        failure_samples = (df['Label'] == 1).sum()
        print(f"\n🏷️  Label Distribution:")
        print(f"   Normal (0):  {(df['Label'] == 0).sum()} samples")
        print(f"   Failure (1): {failure_samples} samples")
        
        if failure_samples == 0:
            print(f"   🚨 CRITICAL: No failure samples!")
    
    # Check for degradation trend
    first_half_mean = df.head(500)['SNR_dB'].mean()
    second_half_mean = df.tail(500)['SNR_dB'].mean()
    degradation = first_half_mean - second_half_mean
    
    print(f"\n📉 Degradation Analysis:")
    print(f"   First 500 samples mean:  {first_half_mean:.2f} dB")
    print(f"   Last 500 samples mean:   {second_half_mean:.2f} dB")
    print(f"   Total degradation:       {degradation:.2f} dB")
    
    if abs(degradation) < 1.0:
        print(f"   ⚠️  WARNING: Very little degradation detected (<1dB)")
    
    # Time to first failure
    failure_indices = df[df['SNR_dB'] < 15.0].index
    if len(failure_indices) > 0:
        first_failure = failure_indices[0]
        print(f"\n⏱️  Failure Timing:")
        print(f"   First failure at: t={first_failure}s ({first_failure/len(df)*100:.1f}% through)")
    else:
        print(f"\n⏱️  Failure Timing:")
        print(f"   No failures detected in entire sequence")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # SNR over time
    ax1.plot(df['Time_Seconds'], df['SNR_dB'], alpha=0.7, linewidth=1)
    ax1.axhline(y=15, color='r', linestyle='--', label='Failure Threshold (15dB)')
    ax1.axhline(y=18, color='orange', linestyle='--', label='Warning Threshold (18dB)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title(f'SNR Evolution: {filename}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SNR histogram
    ax2.hist(df['SNR_dB'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=15, color='r', linestyle='--', label='Failure Threshold')
    ax2.axvline(x=18, color='orange', linestyle='--', label='Warning Threshold')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SNR Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"diagnostics_{filename.replace('.csv', '.png')}"
    plt.savefig(f"results/{plot_filename}", dpi=150)
    print(f"\n📊 Plot saved: results/{plot_filename}")
    plt.close()
    
    return df

def compare_all_datasets():
    """Compare all datasets side by side"""
    datasets = [
        'training_normal.csv',
        'training_ou_failure.csv',
        'training_exp_failure.csv',
        'training_weibull_failure.csv',
        'training_step_failure.csv',
        'training_osc_failure.csv'
    ]
    
    summary = []
    
    for dataset in datasets:
        filepath = os.path.join("training_data", dataset)
        if not os.path.exists(filepath):
            continue
        
        df = pd.read_csv(filepath)
        
        summary.append({
            'Dataset': dataset.replace('training_', '').replace('.csv', ''),
            'Min SNR': f"{df['SNR_dB'].min():.2f}",
            'Max SNR': f"{df['SNR_dB'].max():.2f}",
            'Mean SNR': f"{df['SNR_dB'].mean():.2f}",
            'Below 15dB': f"{(df['SNR_dB'] < 15.0).sum()}",
            'Below 18dB': f"{(df['SNR_dB'] < 18.0).sum()}",
        })
    
    df_summary = pd.DataFrame(summary)
    
    print(f"\n{'='*70}")
    print("DATASET COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"{'='*70}")
    
    return df_summary

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    DATA DIAGNOSTICS TOOL                         ║
║                                                                  ║
║  This tool analyzes your generated datasets to identify issues  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    # Diagnose each dataset
    datasets = [
        'training_normal.csv',
        'training_ou_failure.csv',
        'training_exp_failure.csv',
        'training_weibull_failure.csv',
        'training_step_failure.csv',
        'training_osc_failure.csv'
    ]
    
    for dataset in datasets:
        diagnose_dataset(dataset)
    
    # Overall comparison
    compare_all_datasets()
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
    print("\nCheck the results/ directory for diagnostic plots")

if __name__ == "__main__":
    main()
