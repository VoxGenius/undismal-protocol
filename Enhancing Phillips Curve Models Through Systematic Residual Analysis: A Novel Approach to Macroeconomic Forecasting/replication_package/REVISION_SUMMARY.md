# Summary of Major Revisions

Based on the referee report dated July 29, 2025, the following major revisions have been implemented:

## 1. Fixed Internal Inconsistencies in OOS Improvements
- **Issue**: Conflicting RMSE reduction claims (97-98%, 23.2%, 80-82%)
- **Resolution**: Standardized to 80-82% RMSE reduction in main text, matching Table 5 (tab:oos)
- **Clarification**: Added note that Table 2 (model_comparison) shows 23.2% improvement using full evaluation period with consistent methodology

## 2. Fixed Expectations Variable Measurement
- **Issue**: MICH (Consumer Sentiment) incorrectly used as inflation expectations
- **Resolution**: Changed to MICH1Y (University of Michigan 1-Year Ahead Expected Inflation Rate)
- **Updated**: Variable definitions in both main text and appendix

## 3. Harmonized Sample Periods
- **Issue**: Inconsistent sample period claims (1990-2023 vs 1960-2023)
- **Resolution**: Clarified primary analysis uses 1990-2023
- **Added**: Explanation that enhanced model is further restricted by T5YIE availability (2003+)
- **Noted**: Different observation counts (132 vs 71) due to data constraints

## 4. Documented Real-Time/Vintage Data Usage
- **Added**: New appendix section on "Real-Time Data and Vintage Considerations"
- **Specified**: ALFRED vintage usage for NROU
- **Clarified**: Which variables are subject to revision vs market-based/finalized data
- **Confirmed**: No look-ahead bias in rolling window implementation

## 5. Enhanced Multiple Testing Details
- **Added**: Full breakdown of 89 candidates across 7 domains
- **Specified**: Nested cross-validation procedure (60/20/20 split)
- **Included**: Superior Predictive Ability (SPA) test details
- **Clarified**: Lag specifications and transformations tested

## 6. Added Structural Break Analysis Table
- **Included**: Table with Bai-Perron test results
- **Specified**: Break dates with confidence intervals (1991:Q2, CI: 1990:Q3-1992:Q1)
- **Added**: CUSUM/CUSUM-SQ results and parameter stability measures

## 7. Clarified Forecast Testing Details
- **Added**: Dedicated subsection explaining each test
- **Specified**: Diebold-Mariano with HAC standard errors
- **Clarified**: Hansen-West is Superior Predictive Ability (SPA) test
- **Detailed**: Encompassing test methodology

## 8. Added Recession Prediction Horse-Race
- **New Table**: Comparing PC residuals to standard benchmarks
- **Benchmarks**: 10Y-3M spread, excess bond premium, Sahm rule
- **Results**: PC residuals underperform individually but add value in combined model
- **Statistical Tests**: DeLong tests for AUC differences

## 9. Fixed Minor Issues
- **Table References**: Fixed "Table 3" to proper LaTeX references
- **Clarified**: Table numbering throughout
- **Added**: Reconciliation note for different RMSE improvements

## Additional Improvements
- Added clarification on identification/endogeneity limitations
- Enhanced discussion of the multiple testing vs. economic significance tension
- Improved documentation of replication materials and computational implementation

All revisions maintain the paper's core findings while addressing the referee's methodological concerns and improving clarity.

## Second Round of Major Revisions (Additional Referee Comments)

### High Priority (Blocking) Issues Addressed:

1. **Unified OOS Protocol**
   - Added canonical OOS protocol specification (h=1, 2003-2023, 60-month rolling)
   - Created config/oos_protocol.json with complete specification
   - Clarified Table 2 vs Table 5 differences (full period vs aligned sample)

2. **Fixed Sampling Comparability**
   - Updated Table 5 to show aligned samples (48 predictions for both models)
   - Clarified evaluation window constraints due to T5YIE availability

3. **Fixed Testing Labels**
   - Clarified Hansen-West refers to Hansen (2005) SPA test throughout
   - Added note explaining the labeling in forecast test details

4. **Added Clark-West Test**
   - Added section on Clark-West test for nested models
   - Specified HAC-robust implementation with test statistic (4.82, p<0.01)

5. **Published Full Candidate Ledger**
   - Created outputs/full_candidate_ledger.csv with all 89 variables
   - Included transformations, lags, p-values, OOS improvements, selection decisions

6. **Added Vintage Date Matrix**
   - Created outputs/vintage_date_matrix.csv documenting as-of dates
   - Enhanced real-time section confirming no look-ahead bias
   - Clarified publication lags for each variable

### Medium Priority Issues Addressed:

7. **Clarified Structural Break Frequency**
   - Added note explaining quarterly reporting for break tests
   - Clarified monthly data aggregated to quarters for stability

8. **Discussed Expectations Redundancy**
   - Added new subsection on multicollinearity and VIF analysis
   - Explained complementarity of MICH1Y vs T5YIE (ρ=0.62)
   - Reported principal component robustness check

9. **Ensured Recession Horse-Race Comparability**
   - Added details on real-time implementation for each benchmark
   - Specified data sources and publication lags
   - Confirmed identical train/test periods

### Low Priority Issues Addressed:

10. **Fixed Editing Issues**
    - Updated Table 6 references to use \ref{tab:robustness}
    - Added forecast horizon (h=1) to abstract
    - Fixed all table cross-references

### Additional Deliverables:

- Created reproducibility checklist items:
  - Config file: config/oos_protocol.json
  - Candidate ledger: outputs/full_candidate_ledger.csv  
  - Vintage matrix: outputs/vintage_date_matrix.csv
  - All code, tables, and figures in replication repository