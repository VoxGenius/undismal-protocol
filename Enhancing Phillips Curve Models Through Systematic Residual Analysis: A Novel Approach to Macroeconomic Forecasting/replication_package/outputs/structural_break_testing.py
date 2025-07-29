"""
Structural Break Testing - Critical Requirement #5
Test for structural breaks in Phillips Curve relationship
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt
from fredapi import Fred
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# FRED API setup
fred = Fred(api_key='7a74bbd246e54e7458184b0137db2311')

class StructuralBreakTester:
    def __init__(self):
        self.data = None
        self.baseline_model = None
        self.break_tests = {}
        self.break_dates = []
        
    def load_phillips_curve_data(self):
        """Load Phillips Curve data for structural break testing"""
        
        print("CRITICAL REQUIREMENT #5: STRUCTURAL BREAK TESTING")
        print("=" * 60)
        print("Progress: 62% | Loading Phillips Curve data...")
        
        # Load key Phillips Curve variables
        series_needed = {
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate', 
            'NROU': 'Natural Rate of Unemployment',
            'MICH': 'Michigan Consumer Sentiment'
        }
        
        data_dict = {}
        for code, name in series_needed.items():
            try:
                series = fred.get_series(code, '1970-01-01', '2023-12-31')
                data_dict[code] = series
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                
        self.data = pd.DataFrame(data_dict)
        
        # Create Phillips Curve variables
        self.data['inflation'] = self.data['CPIAUCSL'].pct_change(12) * 100
        self.data['unemployment_gap'] = self.data['UNRATE'] - self.data['NROU']
        self.data['inflation_expectations'] = self.data['MICH']
        
        # Clean data
        required_vars = ['inflation', 'unemployment_gap', 'inflation_expectations']
        self.data = self.data.dropna(subset=required_vars)
        
        print(f"✓ Dataset prepared: {len(self.data)} observations")
        print(f"✓ Sample period: {self.data.index[0].strftime('%Y-%m')} to {self.data.index[-1].strftime('%Y-%m')}")
        
    def fit_baseline_phillips_curve(self):
        """Fit baseline Phillips Curve for break testing"""
        
        print("\\nProgress: 68% | Fitting baseline Phillips Curve...")
        
        # Prepare variables
        y = self.data['inflation']
        X = self.data[['unemployment_gap', 'inflation_expectations']]
        X = sm.add_constant(X)
        
        # Fit baseline model
        self.baseline_model = sm.OLS(y, X).fit()
        
        print(f"✓ Baseline model fitted")
        print(f"  R²: {self.baseline_model.rsquared:.4f}")
        print(f"  Observations: {self.baseline_model.nobs}")
        
    def cusum_test(self):
        """Perform CUSUM test for structural stability"""
        
        print("\\nProgress: 72% | Performing CUSUM test...")
        
        try:
            # CUSUM test for recursive residuals
            cusum_stat, cusum_p = breaks_cusumolsresid(
                self.baseline_model.resid, 
                self.baseline_model.model.exog
            )
            
            self.break_tests['cusum'] = {
                'statistic': cusum_stat,
                'p_value': cusum_p,
                'interpretation': 'Parameter stability test',
                'null_hypothesis': 'Parameters are stable over time',
                'result': 'STABLE' if cusum_p > 0.05 else 'UNSTABLE'
            }
            
            print(f"✓ CUSUM test completed")
            print(f"  Test statistic: {cusum_stat:.4f}")
            print(f"  P-value: {cusum_p:.4f}")
            print(f"  Result: {self.break_tests['cusum']['result']}")
            
        except Exception as e:
            print(f"✗ CUSUM test failed: {e}")
            self.break_tests['cusum'] = {'error': str(e)}
            
    def chow_test_known_breaks(self):
        """Perform Chow tests for known potential break dates"""
        
        print("\\nProgress: 78% | Performing Chow tests for known breaks...")
        
        # Known potential break dates for Phillips Curve
        potential_breaks = [
            ('1979-10', 'Volcker disinflation'),
            ('1990-07', 'Early 1990s recession'),
            ('2001-03', 'Dot-com recession'),
            ('2008-09', 'Great Financial Crisis'),
            ('2020-03', 'COVID-19 pandemic')
        ]
        
        chow_results = {}
        
        for break_date, description in potential_breaks:
            try:
                # Find break point index
                break_date_ts = pd.to_datetime(break_date)
                
                if break_date_ts in self.data.index:
                    break_idx = self.data.index.get_loc(break_date_ts)
                    
                    # Need at least 20% of sample on each side
                    min_size = int(0.2 * len(self.data))
                    
                    if break_idx > min_size and break_idx < (len(self.data) - min_size):
                        
                        # Split sample
                        pre_break = self.data.iloc[:break_idx]
                        post_break = self.data.iloc[break_idx:]
                        
                        # Fit models on subsamples
                        y_pre = pre_break['inflation']
                        X_pre = sm.add_constant(pre_break[['unemployment_gap', 'inflation_expectations']])
                        model_pre = sm.OLS(y_pre, X_pre).fit()
                        
                        y_post = post_break['inflation']
                        X_post = sm.add_constant(post_break[['unemployment_gap', 'inflation_expectations']])
                        model_post = sm.OLS(y_post, X_post).fit()
                        
                        # Calculate Chow test statistic
                        ssr_unrestricted = model_pre.ssr + model_post.ssr
                        ssr_restricted = self.baseline_model.ssr
                        
                        k = len(self.baseline_model.params)  # Number of parameters
                        n = len(self.data)
                        
                        chow_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (n - 2*k))
                        chow_p = 1 - stats.f.cdf(chow_stat, k, n - 2*k)
                        
                        chow_results[break_date] = {
                            'description': description,
                            'statistic': chow_stat,
                            'p_value': chow_p,
                            'pre_break_r2': model_pre.rsquared,
                            'post_break_r2': model_post.rsquared,
                            'result': 'BREAK DETECTED' if chow_p < 0.05 else 'NO BREAK'
                        }
                        
                        print(f"  {break_date} ({description}): F = {chow_stat:.3f}, p = {chow_p:.4f}")
                        
            except Exception as e:
                print(f"  ✗ {break_date}: {e}")
                
        self.break_tests['chow_known'] = chow_results
        
        # Count significant breaks
        significant_breaks = sum(1 for result in chow_results.values() if result['p_value'] < 0.05)
        print(f"✓ Chow tests completed: {significant_breaks}/{len(chow_results)} significant breaks")
        
    def rolling_parameter_analysis(self):
        """Analyze parameter stability using rolling windows"""
        
        print("\\nProgress: 85% | Performing rolling parameter analysis...")
        
        window_size = 60  # 5-year rolling windows
        step_size = 6     # Move window every 6 months
        
        if len(self.data) < window_size * 2:
            print("✗ Insufficient data for rolling analysis")
            return
            
        rolling_results = []
        
        for start_idx in range(0, len(self.data) - window_size, step_size):
            end_idx = start_idx + window_size
            
            window_data = self.data.iloc[start_idx:end_idx]
            window_date = window_data.index[-1]  # End date of window
            
            try:
                # Fit model on window
                y_window = window_data['inflation']
                X_window = sm.add_constant(window_data[['unemployment_gap', 'inflation_expectations']])
                
                window_model = sm.OLS(y_window, X_window).fit()
                
                rolling_results.append({
                    'date': window_date,
                    'const': window_model.params['const'],
                    'unemployment_gap': window_model.params['unemployment_gap'],
                    'inflation_expectations': window_model.params['inflation_expectations'],
                    'r_squared': window_model.rsquared,
                    'observations': len(y_window)
                })
                
            except Exception as e:
                continue
                
        if len(rolling_results) > 0:
            self.rolling_parameters = pd.DataFrame(rolling_results)
            self.rolling_parameters.set_index('date', inplace=True)
            
            # Calculate parameter stability metrics
            param_volatility = {
                'const': self.rolling_parameters['const'].std(),
                'unemployment_gap': self.rolling_parameters['unemployment_gap'].std(),
                'inflation_expectations': self.rolling_parameters['inflation_expectations'].std()
            }
            
            self.break_tests['rolling_analysis'] = {
                'windows_analyzed': len(rolling_results),
                'parameter_volatility': param_volatility,
                'r_squared_range': [
                    self.rolling_parameters['r_squared'].min(),
                    self.rolling_parameters['r_squared'].max()
                ]
            }
            
            print(f"✓ Rolling analysis completed: {len(rolling_results)} windows")
            print(f"  Parameter volatility:")
            for param, vol in param_volatility.items():
                print(f"    {param}: {vol:.4f}")
        else:
            print("✗ No valid rolling windows computed")
            
    def create_break_test_visualizations(self):
        """Create comprehensive structural break visualizations"""
        
        print("\\nProgress: 92% | Creating break test visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Phillips Curve over time with potential breaks
        dates = self.data.index
        inflation = self.data['inflation']
        unemployment_gap = self.data['unemployment_gap']
        
        axes[0,0].plot(dates, inflation, alpha=0.7, label='Inflation', color='red')
        axes[0,0].plot(dates, unemployment_gap, alpha=0.7, label='Unemployment Gap', color='blue')
        
        # Mark potential break dates
        if 'chow_known' in self.break_tests:
            for break_date, result in self.break_tests['chow_known'].items():
                if result['p_value'] < 0.05:
                    break_ts = pd.to_datetime(break_date)
                    axes[0,0].axvline(x=break_ts, color='black', linestyle='--', alpha=0.5)
                    
        axes[0,0].set_title('Phillips Curve Variables with Potential Breaks')
        axes[0,0].set_ylabel('Percent')
        axes[0,0].legend()
        
        # 2. Model residuals over time
        if self.baseline_model:
            residuals = pd.Series(self.baseline_model.resid, index=self.data.index)
            
            axes[0,1].plot(residuals.index, residuals, alpha=0.7, color='green')
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0,1].set_title('Model Residuals Over Time')
            axes[0,1].set_ylabel('Residuals')
            
        # 3. Rolling parameter estimates
        if hasattr(self, 'rolling_parameters'):
            rolling_data = self.rolling_parameters
            
            axes[1,0].plot(rolling_data.index, rolling_data['unemployment_gap'], 
                          label='Unemployment Gap Coef', color='blue')
            axes[1,0].plot(rolling_data.index, rolling_data['inflation_expectations'], 
                          label='Expectations Coef', color='red')
            axes[1,0].set_title('Rolling Parameter Estimates')
            axes[1,0].set_ylabel('Coefficient Value')
            axes[1,0].legend()
            
            # 4. Rolling R-squared
            axes[1,1].plot(rolling_data.index, rolling_data['r_squared'], 
                          color='purple', linewidth=2)
            axes[1,1].set_title('Rolling R-squared')
            axes[1,1].set_ylabel('R-squared')
            axes[1,1].set_ylim([0, 1])
        else:
            axes[1,0].text(0.5, 0.5, 'Rolling parameter\\nanalysis not available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,1].text(0.5, 0.5, 'Rolling R-squared\\nanalysis not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig('charts/structural_break_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Break test visualizations created")
        
    def save_break_test_results(self):
        """Save all structural break test results"""
        
        # Save break test results
        import json
        with open('outputs/structural_break_tests.json', 'w') as f:
            json.dump(self.break_tests, f, indent=2, default=str)
            
        # Save rolling parameters if available
        if hasattr(self, 'rolling_parameters'):
            self.rolling_parameters.to_csv('outputs/rolling_parameter_estimates.csv')
            
        # Generate break test report
        with open('outputs/structural_break_report.txt', 'w') as f:
            f.write("STRUCTURAL BREAK TEST REPORT\\n")
            f.write("="*40 + "\\n\\n")
            
            f.write(f"Sample Period: {self.data.index[0].strftime('%Y-%m')} to {self.data.index[-1].strftime('%Y-%m')}\\n")
            f.write(f"Observations: {len(self.data)}\\n\\n")
            
            # CUSUM test results
            if 'cusum' in self.break_tests:
                cusum = self.break_tests['cusum']
                if 'error' not in cusum:
                    f.write("CUSUM TEST RESULTS:\\n")
                    f.write(f"Test Statistic: {cusum['statistic']:.4f}\\n")
                    f.write(f"P-value: {cusum['p_value']:.4f}\\n")
                    f.write(f"Result: {cusum['result']}\\n\\n")
                    
            # Chow test results
            if 'chow_known' in self.break_tests:
                f.write("CHOW TEST RESULTS:\\n")
                f.write(f"{'Date':<10} {'Event':<25} {'F-stat':<8} {'P-value':<8} {'Result'}\\n")
                f.write("-" * 70 + "\\n")
                
                for break_date, result in self.break_tests['chow_known'].items():
                    f.write(f"{break_date:<10} {result['description'][:24]:<25} ")
                    f.write(f"{result['statistic']:<8.3f} {result['p_value']:<8.4f} {result['result']}\\n")
                    
            # Rolling analysis summary
            if 'rolling_analysis' in self.break_tests:
                rolling = self.break_tests['rolling_analysis']
                f.write("\\nROLLING PARAMETER ANALYSIS:\\n")
                f.write(f"Windows Analyzed: {rolling['windows_analyzed']}\\n")
                f.write(f"R-squared Range: {rolling['r_squared_range'][0]:.3f} - {rolling['r_squared_range'][1]:.3f}\\n")
                
        print("✓ Structural break test results saved")
        
        # Print summary
        print("\\n" + "="*60)
        print("STRUCTURAL BREAK TESTING COMPLETE:")
        print("="*60)
        
        if 'cusum' in self.break_tests and 'error' not in self.break_tests['cusum']:
            cusum_result = self.break_tests['cusum']['result']
            print(f"CUSUM test: {cusum_result}")
            
        if 'chow_known' in self.break_tests:
            chow_results = self.break_tests['chow_known']
            significant_breaks = sum(1 for r in chow_results.values() if r['p_value'] < 0.05)
            print(f"Significant structural breaks: {significant_breaks}/{len(chow_results)}")
            
        if 'rolling_analysis' in self.break_tests:
            rolling = self.break_tests['rolling_analysis']
            print(f"Rolling windows analyzed: {rolling['windows_analyzed']}")
            
        print("✓ Phillips Curve stability comprehensively assessed")

if __name__ == "__main__":
    
    tester = StructuralBreakTester()
    
    # Execute structural break testing
    tester.load_phillips_curve_data()
    tester.fit_baseline_phillips_curve()
    tester.cusum_test()
    tester.chow_test_known_breaks()
    tester.rolling_parameter_analysis()
    tester.create_break_test_visualizations()
    tester.save_break_test_results()
    
    print("\\n✓ CRITICAL REQUIREMENT #5 COMPLETE: STRUCTURAL BREAKS TESTED")
    print("Progress: 100% | Estimated remaining time: 2-3 hours")