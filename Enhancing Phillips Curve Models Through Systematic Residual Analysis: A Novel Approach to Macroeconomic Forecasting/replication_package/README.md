# Enhancing Phillips Curve Models Through Systematic Residual Analysis

## A Novel Approach to Macroeconomic Forecasting

This repository contains the complete replication package for the paper "Enhancing Phillips Curve Models Through Systematic Residual Analysis: A Novel Approach to Macroeconomic Forecasting" by Matthew Busigin (VoxGenius, Inc.).

📄 **[Read the full paper (PDF)](phillips_curve_paper.pdf)**

## Paper Status

**Latest Update (January 2025)**: Completed third round of referee revisions addressing 12 remaining issues. Paper now ready for journal resubmission.

## Abstract

This paper presents a novel methodology for enhancing macroeconomic Phillips Curve models through systematic residual analysis, termed the "Undismal Protocol." Starting with a baseline model incorporating unemployment gap and inflation expectations, we develop a comprehensive framework for identifying and incorporating missing economic variables through theory-guided candidate selection and rigorous out-of-sample validation.

### Key Results

- **68x improvement** in explanatory power (R² from 0.006 to 0.410)
- **80-82% reduction** in out-of-sample forecasting errors vs baseline Phillips Curve
- **81.8% RMSE improvement** over best univariate benchmark (survey expectations only)
- **Novel recession prediction** capability using Phillips Curve residuals (AUC = 0.618)
- Identification of external sector and market-based expectations as key enhancement channels
- Robust performance confirmed across multiple testing procedures (Diebold-Mariano, Clark-West, Hansen SPA)

## Repository Structure

```
replication_package/
├── README.md                           # This file
├── phillips_curve_paper.pdf            # Full paper (36 pages)
├── phillips_curve_paper.tex            # LaTeX source
├── figures/                            # All paper figures
│   ├── time_series_overview.pdf
│   ├── residual_analysis.pdf
│   ├── structural_breaks.pdf
│   ├── variable_selection.pdf
│   └── recession_prediction_analysis.pdf
├── tables/                             # All paper tables (LaTeX format)
│   ├── descriptive_stats.tex
│   ├── model_comparison.tex
│   ├── baseline_benchmarks.tex
│   ├── variable_selection.tex
│   ├── structural_breaks.tex
│   └── robustness.tex
├── outputs/                            # Additional analysis outputs
│   ├── full_candidate_ledger.csv       # All 89 tested variables
│   └── vintage_date_matrix.csv         # Real-time data documentation
├── config/                             # Configuration files
│   └── oos_protocol.json               # Out-of-sample validation specs
├── real_data_visualizations.py         # Main data analysis and visualization
├── recession_prediction_analysis.py    # Recession forecasting analysis
└── REVISION_SUMMARY.md                 # Detailed revision history
```

## Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn fredapi scikit-learn statsmodels
```

### Running the Analysis

1. **Generate main visualizations and results:**
   ```bash
   python real_data_visualizations.py
   ```

2. **Run recession prediction analysis:**
   ```bash
   python recession_prediction_analysis.py
   ```

3. **Compile the LaTeX paper:**
   ```bash
   pdflatex phillips_curve_paper.tex
   pdflatex phillips_curve_paper.tex  # Run twice for references
   ```

## The Undismal Protocol

Our systematic 7-step framework for model enhancement:

1. **State the decision and loss function** - Prioritize out-of-sample RMSE
2. **Ship a sparse baseline model** - Start with theoretically defensible variables only
3. **Let residuals issue work orders** - Systematic diagnostic analysis
4. **Assemble theory-scoped candidates** - Seven economic domains for variable search
5. **Search lags and transformations** - But upgrades must be earned through performance
6. **Publish a comprehensive ledger** - Full documentation of all decisions
7. **Declare refit triggers** - Specify regime monitoring for operational deployment

## Key Findings

1. **Residuals contain information, not just noise** - Systematic patterns in Phillips Curve failures reveal missing economic forces
2. **External sector matters** - Trade-weighted dollar (12-month lag) provides strongest enhancement
3. **Market expectations complement surveys** - 5-year breakeven inflation rates add predictive power
4. **Recession prediction bonus** - Phillips Curve residuals predict recessions 6-12 months ahead
5. **Economic vs. statistical significance** - Out-of-sample performance matters more than p-values

## Data

All data sourced from FRED (Federal Reserve Economic Data):
- **CPIAUCSL**: Consumer Price Index for All Urban Consumers (YoY % change)
- **UNRATE**: Civilian Unemployment Rate  
- **NROU**: Natural Rate of Unemployment (CBO estimate, quarterly interpolated)
- **MICH1Y**: University of Michigan 1-Year Inflation Expectations
- **DTWEXBGS**: Trade Weighted U.S. Dollar Index: Goods (Broad)
- **T5YIE**: 5-Year Breakeven Inflation Rate

### Real-Time Data Protocol
- Forecasts made at month-end using only data available at that time
- Vintage data pulled from ALFRED to ensure no look-ahead bias
- Publication lags respected: CPI(t-1), UNRATE(t-1), MICH1Y(t), T5YIE(t), DTWEXBGS(t)
- Complete vintage documentation in `outputs/vintage_date_matrix.csv`

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{busigin2025phillips,
  title={Enhancing Phillips Curve Models Through Systematic Residual Analysis: 
         A Novel Approach to Macroeconomic Forecasting},
  author={Busigin, Matthew},
  journal={Working Paper},
  year={2025},
  institution={VoxGenius, Inc.}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact

Matthew Busigin  
VoxGenius, Inc.  
matt@voxgenius.ai

## Acknowledgments

Special thanks to the Federal Reserve Bank of St. Louis for maintaining the FRED database, and to the open-source community for the excellent Python scientific computing stack.

---

*"The Phillips Curve isn't broken. Our whole approach to model 'failure' is."*