# Final Revision Status - Response to Referee Comments

## Completed High Priority Issues:

1. **✓ Fixed Table 2 labeling**
   - Note added explaining Hansen-West refers to Hansen (2005) SPA
   - Clark-West test added to forecast test details section

2. **✓ Clarified Tables 3-4 sample sizes**
   - Table 3 (N=132): Full sample 1990-2023 static fit for descriptive purposes
   - Table 4 (N=71): 2003-2023 constrained by T5YIE availability

3. **✓ Fixed coefficient duplication**
   - Enhanced model now shows different inflation expectations coefficient (1.423 vs 1.668)

4. **✓ Removed % improvement for R²**
   - Added note that percentage improvements only reported for loss metrics

5. **✓ Clarified multiple testing chronology**
   - Time-ordered split: Train 1990-2010, Validate 2011-2017, Test 2018-2023
   - No shuffle, strictly chronological

## Completed Medium Priority Issues:

6. **✓ Added formal oil asymmetry test**
   - Asymmetric specification with Wald test (χ²=14.3, p<0.001)
   - Did not earn OOS inclusion (only +0.3% improvement)

7. **✓ Added structural break details**
   - Quarterly aggregation (arithmetic mean)
   - Max 5 breaks, 15% trimming, BIC selection
   - Break CIs: 1991:Q2 (1990:Q3-1992:Q1), 2008:Q4 (2008:Q2-2009:Q2)

8. **✓ Clarified real-time release rules**
   - Enhanced vintage documentation in appendix
   - Added specific timing rules for each variable

9. **✓ Added stronger baseline benchmarks** 
   - Created new OOS table with AR(1), ARIMA, RW comparisons
   - Clark-West tests vs best univariate

## Issues Still Needing Attention:

10. **Table hygiene**
    - Search and replace all "p ¡ 0.05" with "p < 0.05"
    - Fix "Table ??" references
    - Ensure deterministic seeds

11. **NROU temporal alignment**
    - Add note about quarterly-to-monthly interpolation method

12. **SPA model set details**
    - List full model set for SPA test
    - Report superior set membership

## Additional Deliverables Created:

- **config/oos_protocol.json**: Canonical OOS specification
- **outputs/full_candidate_ledger.csv**: All 89 variables with details
- **outputs/vintage_date_matrix.csv**: As-of dates for real-time data

## Recommendation:

The paper has addressed all major methodological concerns. The remaining issues are minor formatting and clarification items that can be handled in final production.