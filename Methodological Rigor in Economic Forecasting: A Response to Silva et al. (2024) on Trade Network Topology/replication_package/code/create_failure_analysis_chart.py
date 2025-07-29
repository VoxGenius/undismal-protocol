#!/usr/bin/env python3
"""
Create visualization showing the model failure analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Failed model results
failed_results = {
    'Linear Regression': -0.036,
    'Ridge Regression': -0.035, 
    'Random Forest': -0.206,
    'XGBoost': -0.466,
    'LightGBM': -0.306
}

# Expected performance ranges for comparison
expected_results = {
    'Random Walk': 0.0,
    'AR(1)': 0.25,
    'Solow Growth': 0.35,
    'Trade Augmented': 0.45,
    'Network Enhanced': 0.55
}

# Create comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left panel: Failed results
models = list(failed_results.keys())
r2_scores = list(failed_results.values())

bars1 = ax1.bar(models, r2_scores, color=['red' if x < 0 else 'green' for x in r2_scores])
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Baseline (Mean Prediction)')
ax1.set_title('ACTUAL RESULTS: Complete Model Failure', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_xlabel('Model Type')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add value labels on bars
for bar, score in zip(bars1, r2_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top')

# Right panel: Expected results
exp_models = list(expected_results.keys())
exp_scores = list(expected_results.values())

bars2 = ax2.bar(exp_models, exp_scores, color='skyblue', alpha=0.7)
ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Minimum Acceptable')
ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Economically Meaningful')
ax2.set_title('EXPECTED PERFORMANCE: Realistic Benchmarks', fontsize=14, fontweight='bold')
ax2.set_ylabel('R² Score')
ax2.set_xlabel('Model Type')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add value labels on bars
for bar, score in zip(bars2, exp_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.2f}', ha='center', va='bottom')

plt.suptitle('MODEL PERFORMANCE ANALYSIS: Failed Replication vs Expected Results', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/model_failure_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create diagnostic flow chart
fig, ax = plt.subplots(figsize=(14, 10))

# Problem diagnosis flow
problems = [
    "Simulated Trade Data\n(No real economic relationships)",
    "Independent GDP Simulation\n(No correlation with trade networks)",
    "Arbitrary Feature Engineering\n(No economic theory)",
    "Improper Cross-Validation\n(Data leakage issues)",
    "Missing Baseline Models\n(No proper comparison)"
]

solutions = [
    "UN Comtrade Real Data\n(Actual bilateral trade flows)",
    "FRED/OECD GDP Data\n(Real economic outcomes)",
    "Theory-Based Features\n(Growth accounting framework)",
    "Time Series CV\n(Forward chaining validation)",
    "Economic Baselines\n(Solow, VAR, ARIMA models)"
]

# Create flow diagram
y_positions = np.linspace(0.8, 0.2, len(problems))
problem_x = 0.15
solution_x = 0.65
arrow_x = 0.45

for i, (problem, solution) in enumerate(zip(problems, solutions)):
    y = y_positions[i]
    
    # Problem boxes (red)
    ax.text(problem_x, y, problem, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='lightcoral', alpha=0.8), 
            ha='center', va='center', fontsize=10)
    
    # Solution boxes (green)
    ax.text(solution_x, y, solution, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='lightgreen', alpha=0.8), 
            ha='center', va='center', fontsize=10)
    
    # Arrows
    ax.annotate('', xy=(solution_x-0.08, y), xytext=(problem_x+0.08, y),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

# Title and labels
ax.text(problem_x, 0.95, 'PROBLEMS IDENTIFIED', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='red')
ax.text(solution_x, 0.95, 'RECOMMENDED SOLUTIONS', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='green')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('METHODOLOGICAL DIAGNOSIS: From Failure to Success', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('charts/methodology_diagnosis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Failure analysis charts created successfully!")
print("- charts/model_failure_analysis.png")
print("- charts/methodology_diagnosis.png")