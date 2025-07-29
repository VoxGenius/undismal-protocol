# Step 1: Decision & Loss Function

## The Decision
We aim to explain Phillips Curve residuals using systematic variable addition based on Out-of-Sample (OOS) Root Mean Square Error (RMSE) using ALFRED real-time data.

## Loss Function
**Primary Metric**: OOS RMSE calculated using ALFRED real-time data vintages
- **Target**: Minimize forecast errors in inflation dynamics
- **Evaluation**: Rolling window cross-validation with real-time data constraints
- **Benchmark**: Current Phillips Curve model performance

## Current Baseline Performance
Based on related research findings:
- **Baseline R²**: ~65-75% (typical Phillips Curve models)
- **Enhancement Target**: Achieve >90% explanatory power
- **Error Reduction Goal**: >40% RMSE improvement

## Addition Criteria
Variables join the model **only if** they:
1. Reduce OOS RMSE on ALFRED data
2. Pass statistical significance tests (p < 0.05)
3. Have clear economic theoretical justification
4. Maintain model parsimony (avoid overfitting)

## Time Horizon
- **Training Window**: 20 years rolling
- **Forecast Horizon**: 1-4 quarters ahead
- **Evaluation Period**: 2000-2023 (ALFRED availability)

*Completion: Step 1 | Progress: 20%*