"""
Theory-Scoped Candidate Assembly - Undismal Protocol Step 4
Assemble theory‑scoped candidates across economic domains
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FRED API setup
fred = Fred(api_key='7a74bbd246e54e7458184b0137db2311')

class CandidateAssembler:
    def __init__(self):
        self.candidates = {}
        self.candidate_data = pd.DataFrame()
        self.theory_domains = [
            'MONETARY', 'FISCAL', 'LABOR_HOURS', 'EXPECTATIONS', 
            'DEMOGRAPHICS', 'FINANCIAL', 'EXTERNAL'
        ]
        
    def define_candidate_universe(self):
        """Define theory-scoped candidate variables by domain"""
        
        self.candidates = {
            'MONETARY': {
                'FEDFUNDS': {'name': 'Federal Funds Rate', 'transform': 'level'},
                'GS10': {'name': '10-Year Treasury Rate', 'transform': 'level'},
                'GS2': {'name': '2-Year Treasury Rate', 'transform': 'level'},
                'DFEDTARU': {'name': 'Taylor Rule Deviation', 'transform': 'level'},
                'M2SL': {'name': 'M2 Money Supply', 'transform': 'yoy_growth'},
                'BOGMBASE': {'name': 'Monetary Base', 'transform': 'yoy_growth'}
            },
            
            'FISCAL': {
                'FGEXPND': {'name': 'Federal Government Expenditures', 'transform': 'yoy_growth'},
                'FGRECPT': {'name': 'Federal Government Receipts', 'transform': 'yoy_growth'},
                'GFDEBTN': {'name': 'Federal Debt to GDP', 'transform': 'level'},
                'FYONGDA188S': {'name': 'Federal Budget Balance/GDP', 'transform': 'level'}
            },
            
            'LABOR_HOURS': {
                'AWHMAN': {'name': 'Avg Weekly Hours Manufacturing', 'transform': 'level'},
                'AWHI': {'name': 'Avg Weekly Hours All Industries', 'transform': 'level'},
                'HOANBS': {'name': 'Hours of All Persons', 'transform': 'yoy_growth'},
                'OPHNFB': {'name': 'Output Per Hour Nonfarm Business', 'transform': 'yoy_growth'}
            },
            
            'EXPECTATIONS': {
                'MICH': {'name': 'Michigan Consumer Sentiment', 'transform': 'level'},
                'T5YIE': {'name': '5-Year Breakeven Inflation Rate', 'transform': 'level'},
                'T10YIE': {'name': '10-Year Breakeven Inflation Rate', 'transform': 'level'},
                'STLFSI4': {'name': 'Financial Stress Index', 'transform': 'level'}
            },
            
            'DEMOGRAPHICS': {
                'CIVPART': {'name': 'Labor Force Participation Rate', 'transform': 'level'},
                'EMRATIO': {'name': 'Employment-Population Ratio', 'transform': 'level'},
                'LNU01300060': {'name': 'Labor Force 25-54 Years', 'transform': 'yoy_growth'}
            },
            
            'FINANCIAL': {
                'DEXUSEU': {'name': 'US/Euro Exchange Rate', 'transform': 'yoy_growth'},
                'VIXCLS': {'name': 'VIX Volatility Index', 'transform': 'level'},
                'DSPIC96': {'name': 'Real Disposable Personal Income', 'transform': 'yoy_growth'},
                'PSAVERT': {'name': 'Personal Saving Rate', 'transform': 'level'}
            },
            
            'EXTERNAL': {
                'DCOILWTICO': {'name': 'WTI Crude Oil Price', 'transform': 'yoy_growth'},
                'GOLDAMGBD228NLBM': {'name': 'Gold Price', 'transform': 'yoy_growth'},
                'DTWEXBGS': {'name': 'Trade Weighted Dollar Index', 'transform': 'yoy_growth'},
                'NETEXP': {'name': 'Net Exports/GDP', 'transform': 'level'}
            }
        }
        
        print("✓ Defined candidate universe across 7 theory domains")
        total_candidates = sum(len(domain) for domain in self.candidates.values())
        print(f"✓ Total candidates: {total_candidates}")
        
    def fetch_candidate_data(self, start_date='1990-01-01', end_date='2023-12-31'):
        """Fetch all candidate variable data"""
        
        print(f"\nFetching candidate data from FRED...")
        
        all_data = {}
        failed_series = []
        
        for domain, variables in self.candidates.items():
            print(f"\n{domain} Variables:")
            
            for code, info in variables.items():
                try:
                    series = fred.get_series(code, start_date, end_date)
                    all_data[code] = series
                    print(f"  ✓ {info['name']}")
                except Exception as e:
                    failed_series.append(code)
                    print(f"  ✗ {info['name']}: {str(e)[:50]}...")
                    
        self.candidate_data = pd.DataFrame(all_data)
        
        print(f"\\n✓ Successfully fetched {len(all_data)} series")
        if failed_series:
            print(f"✗ Failed to fetch {len(failed_series)} series: {', '.join(failed_series)}")
            
        return self.candidate_data
    
    def apply_transformations(self):
        """Apply specified transformations to candidate variables"""
        
        print("\\nApplying transformations...")
        
        transformed_data = {}
        
        for domain, variables in self.candidates.items():
            for code, info in variables.items():
                if code not in self.candidate_data.columns:
                    continue
                    
                series = self.candidate_data[code].copy()
                transform = info['transform']
                
                if transform == 'level':
                    transformed_data[f"{code}_level"] = series
                    
                elif transform == 'yoy_growth':
                    yoy_growth = series.pct_change(12) * 100
                    transformed_data[f"{code}_yoy"] = yoy_growth
                    
                elif transform == 'diff':
                    diff = series.diff()
                    transformed_data[f"{code}_diff"] = diff
                    
        # Also create lagged versions
        lag_periods = [1, 3, 6, 12]
        
        for var_name, series in transformed_data.copy().items():
            for lag in lag_periods:
                lagged_series = series.shift(lag)
                transformed_data[f"{var_name}_lag{lag}"] = lagged_series
                
        self.candidate_data = pd.DataFrame(transformed_data)
        
        print(f"✓ Created {len(self.candidate_data.columns)} transformed variables")
        print(f"✓ Including lags: {len(lag_periods)} periods each")
        
    def test_candidates_against_residuals(self):
        """Test each candidate against baseline residuals"""
        
        # Load baseline residuals
        residuals_data = pd.read_csv('outputs/baseline_residuals.csv')
        residuals_data['date'] = pd.to_datetime(residuals_data['date'])
        
        # Align candidate data with residuals
        aligned_data = self.candidate_data.copy()
        aligned_data.index = pd.to_datetime(aligned_data.index)
        
        # Merge on dates
        merged_data = residuals_data.set_index('date').join(aligned_data, how='inner')
        
        print(f"\\nTesting {len(aligned_data.columns)} candidates against residuals...")
        
        # Test each candidate
        import statsmodels.api as sm
        
        results = []
        
        for var_name in aligned_data.columns:
            if var_name in merged_data.columns and not merged_data[var_name].isna().all():
                
                # Prepare data
                y = merged_data['residuals'].dropna()
                x = merged_data[var_name].reindex(y.index).dropna()
                
                # Align y and x
                common_idx = y.index.intersection(x.index)
                if len(common_idx) > 20:  # Minimum observations
                    
                    y_aligned = y.loc[common_idx]
                    x_aligned = x.loc[common_idx]
                    x_aligned = sm.add_constant(x_aligned)
                    
                    try:
                        model = sm.OLS(y_aligned, x_aligned).fit()
                        
                        # Extract domain from variable name
                        domain = self._get_variable_domain(var_name)
                        
                        results.append({
                            'variable': var_name,
                            'domain': domain,
                            'r_squared': model.rsquared,
                            'p_value': model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan,
                            'coefficient': model.params.iloc[1] if len(model.params) > 1 else np.nan,
                            'observations': len(common_idx),
                            'economic_rationale': self._get_economic_rationale(var_name, domain)
                        })
                        
                    except:
                        continue
                        
        # Sort by R-squared
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('r_squared', ascending=False)
        
        # Save results
        results_df.to_csv('outputs/candidate_test_results.csv', index=False)
        
        print(f"✓ Tested candidates, saved results")
        print(f"✓ Top 5 candidates by R²:")
        
        for i, row in results_df.head().iterrows():
            print(f"  {row['variable']}: R² = {row['r_squared']:.3f}, p = {row['p_value']:.3f}")
            
        return results_df
    
    def _get_variable_domain(self, var_name):
        """Extract domain from variable name"""
        var_code = var_name.split('_')[0]
        
        for domain, variables in self.candidates.items():
            if var_code in variables:
                return domain
        return 'UNKNOWN'
    
    def _get_economic_rationale(self, var_name, domain):
        """Provide economic rationale for variable"""
        
        rationales = {
            'MONETARY': 'Monetary policy transmission affects inflation expectations and demand',
            'FISCAL': 'Fiscal policy influences aggregate demand and inflation pressures',
            'LABOR_HOURS': 'Labor market intensive margins capture capacity utilization effects',
            'EXPECTATIONS': 'Forward-looking expectations drive inflation dynamics',
            'DEMOGRAPHICS': 'Demographic shifts alter labor supply and demand patterns',
            'FINANCIAL': 'Financial conditions transmit to real economy and prices',
            'EXTERNAL': 'External factors influence domestic inflation through trade and commodities'
        }
        
        return rationales.get(domain, 'Unknown economic channel')
    
    def create_candidate_summary(self, results_df):
        """Create comprehensive candidate summary"""
        
        # Domain performance
        domain_performance = results_df.groupby('domain').agg({
            'r_squared': ['count', 'mean', 'max'],
            'p_value': lambda x: (x < 0.05).sum()
        }).round(3)
        
        print("\\n" + "="*60)
        print("CANDIDATE PERFORMANCE BY DOMAIN")
        print("="*60)
        print(domain_performance)
        
        # Top candidates overall
        top_candidates = results_df[results_df['p_value'] < 0.05].head(10)
        
        print("\\n" + "="*60)
        print("TOP 10 STATISTICALLY SIGNIFICANT CANDIDATES")
        print("="*60)
        
        for i, row in top_candidates.iterrows():
            print(f"{i+1:2d}. {row['variable']:<25} (R² = {row['r_squared']:.3f}, p = {row['p_value']:.3f})")
            print(f"     Domain: {row['domain']}")
            print(f"     Rationale: {row['economic_rationale']}")
            print()
            
        # Save summary
        summary = {
            'total_candidates_tested': len(results_df),
            'significant_candidates': len(results_df[results_df['p_value'] < 0.05]),
            'top_r_squared': results_df['r_squared'].max(),
            'domains_tested': results_df['domain'].nunique(),
            'avg_r_squared_by_domain': results_df.groupby('domain')['r_squared'].mean().to_dict()
        }
        
        with open('outputs/candidate_summary.txt', 'w') as f:
            f.write("CANDIDATE ASSEMBLY SUMMARY\\n")
            f.write("="*50 + "\\n\\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\\n")
                
        return summary

if __name__ == "__main__":
    print("UNDISMAL PROTOCOL - STEP 4: THEORY-SCOPED CANDIDATES")
    print("=" * 65)
    
    assembler = CandidateAssembler()
    
    # Define candidate universe
    assembler.define_candidate_universe()
    
    # Fetch data
    assembler.fetch_candidate_data()
    
    # Apply transformations
    assembler.apply_transformations()
    
    # Test against residuals
    results = assembler.test_candidates_against_residuals()
    
    # Create summary
    summary = assembler.create_candidate_summary(results)
    
    print("\\n✓ STEP 4 COMPLETE: Theory-scoped candidates assembled and tested")
    print(f"✓ {summary['significant_candidates']} significant candidates identified")
    print("Next: Lag & transform search with earned upgrades")