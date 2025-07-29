"""
Robustness Checks & Sensitivity Analysis - Critical Requirement #6
Implement comprehensive robustness checks and sensitivity analysis
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from fredapi import Fred
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# FRED API setup
fred = Fred(api_key='7a74bbd246e54e7458184b0137db2311')

class RobustnessAnalyzer:
    def __init__(self):
        self.data = None
        self.robustness_results = {}
        
    def load_comprehensive_data(self):
        """Load comprehensive dataset for robustness analysis"""
        
        print("CRITICAL REQUIREMENT #6: ROBUSTNESS & SENSITIVITY ANALYSIS")
        print("=" * 70)
        print("Progress: 77% | Loading comprehensive dataset...")
        
        # Load all variables used in previous analysis
        series_needed = {
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate', 
            'NROU': 'Natural Rate of Unemployment',
            'MICH': 'Michigan Consumer Sentiment',
            'DTWEXBGS': 'Trade Weighted Dollar Index',
            'T5YIE': '5-Year Breakeven Inflation Rate',
            'STLFSI4': 'Financial Stress Index',
            'GS10': '10-Year Treasury Rate',
            'AWHMAN': 'Average Weekly Hours Manufacturing'
        }
        
        data_dict = {}
        for code, name in series_needed.items():
            try:
                series = fred.get_series(code, '1990-01-01', '2023-12-31')
                data_dict[code] = series
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                continue
                
        self.data = pd.DataFrame(data_dict)
        
        # Create all model variables
        self.prepare_all_variables()
        print(f"✓ Dataset prepared: {len(self.data)} observations")
        
    def prepare_all_variables(self):
        """Prepare all variables and transformations"""
        
        # Core variables
        self.data['inflation'] = self.data['CPIAUCSL'].pct_change(12) * 100
        self.data['unemployment_gap'] = self.data['UNRATE'] - self.data['NROU']
        self.data['inflation_expectations'] = self.data['MICH']
        
        # Enhanced variables
        self.data['dollar_yoy'] = self.data['DTWEXBGS'].pct_change(12) * 100
        self.data['dollar_yoy_lag12'] = self.data['dollar_yoy'].shift(12)
        
        self.data['breakeven_5y'] = self.data['T5YIE']
        self.data['breakeven_5y_lag3'] = self.data['breakeven_5y'].shift(3)
        
        self.data['financial_stress'] = self.data['STLFSI4']
        self.data['financial_stress_lag1'] = self.data['financial_stress'].shift(1)
        
        # Additional robustness variables
        self.data['treasury_10y'] = self.data['GS10']
        self.data['weekly_hours_mfg'] = self.data['AWHMAN']
        
        # Clean data
        required_vars = ['inflation', 'unemployment_gap', 'inflation_expectations']
        self.data = self.data.dropna(subset=required_vars)
        
    def sample_period_sensitivity(self):
        """Test sensitivity to different sample periods"""
        
        print("\\nProgress: 82% | Testing sample period sensitivity...")
        
        # Define different sample periods
        sample_periods = [
            ('1990-01-01', '2019-12-31', 'Pre-COVID'),
            ('1995-01-01', '2023-12-31', 'Post-1995'),
            ('2000-01-01', '2023-12-31', 'Post-2000'),
            ('1990-01-01', '2007-12-31', 'Pre-Crisis'),
            ('2010-01-01', '2023-12-31', 'Post-Crisis')
        ]
        
        sample_results = {}
        baseline_vars = ['unemployment_gap', 'inflation_expectations']
        enhanced_vars = baseline_vars + ['dollar_yoy_lag12', 'breakeven_5y_lag3']
        
        for start_date, end_date, label in sample_periods:
            
            # Filter data for sample period
            sample_data = self.data[start_date:end_date].copy()
            
            if len(sample_data) < 30:  # Need minimum observations
                continue
                
            try:
                # Baseline model
                y = sample_data['inflation'].dropna()
                X_baseline = sample_data[baseline_vars].reindex(y.index).dropna()
                common_idx = y.index.intersection(X_baseline.index)
                
                if len(common_idx) >= 20:
                    y_aligned = y.loc[common_idx]
                    X_aligned = sm.add_constant(X_baseline.loc[common_idx])
                    baseline_model = sm.OLS(y_aligned, X_aligned).fit()
                    
                    # Enhanced model  
                    X_enhanced = sample_data[enhanced_vars].reindex(y.index).dropna()
                    enhanced_common_idx = y.index.intersection(X_enhanced.index)
                    
                    if len(enhanced_common_idx) >= 20:
                        y_enhanced = y.loc[enhanced_common_idx]
                        X_enhanced_aligned = sm.add_constant(X_enhanced.loc[enhanced_common_idx])
                        enhanced_model = sm.OLS(y_enhanced, X_enhanced_aligned).fit()
                        
                        sample_results[label] = {
                            'observations': len(common_idx),
                            'baseline_r2': baseline_model.rsquared,
                            'enhanced_r2': enhanced_model.rsquared,
                            'improvement': enhanced_model.rsquared - baseline_model.rsquared,
                            'baseline_aic': baseline_model.aic,
                            'enhanced_aic': enhanced_model.aic
                        }
                        
                        print(f"  {label}: R² {baseline_model.rsquared:.3f} → {enhanced_model.rsquared:.3f}")
                        
            except Exception as e:
                print(f"  ✗ {label}: {e}")
                continue
                
        self.robustness_results['sample_sensitivity'] = sample_results
        print(f"✓ Sample sensitivity analysis: {len(sample_results)} periods tested")
        
    def variable_specification_sensitivity(self):
        """Test sensitivity to different variable specifications"""
        
        print("\\nProgress: 87% | Testing variable specification sensitivity...")
        
        # Define alternative specifications
        specifications = {
            'baseline': ['unemployment_gap', 'inflation_expectations'],
            'with_dollar': ['unemployment_gap', 'inflation_expectations', 'dollar_yoy_lag12'],
            'with_expectations': ['unemployment_gap', 'inflation_expectations', 'breakeven_5y_lag3'],
            'with_financial': ['unemployment_gap', 'inflation_expectations', 'financial_stress_lag1'],
            'full_enhanced': ['unemployment_gap', 'inflation_expectations', 'dollar_yoy_lag12', 'breakeven_5y_lag3'],
            'alternative_enhanced': ['unemployment_gap', 'inflation_expectations', 'treasury_10y', 'weekly_hours_mfg']
        }
        
        spec_results = {}
        
        for spec_name, variables in specifications.items():
            
            try:
                # Check if all variables are available
                if not all(var in self.data.columns for var in variables):
                    continue
                    
                y = self.data['inflation'].dropna()
                X = self.data[variables].dropna()
                
                # Align data
                common_idx = y.index.intersection(X.index)
                
                if len(common_idx) >= 30:
                    y_aligned = y.loc[common_idx]
                    X_aligned = sm.add_constant(X.loc[common_idx])
                    
                    model = sm.OLS(y_aligned, X_aligned).fit()
                    
                    spec_results[spec_name] = {
                        'variables': variables,
                        'observations': len(common_idx),
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'aic': model.aic,
                        'bic': model.bic,
                        'rmse': np.sqrt(mean_squared_error(y_aligned, model.fittedvalues))
                    }
                    
                    print(f"  {spec_name}: R² = {model.rsquared:.3f}, AIC = {model.aic:.1f}")
                    
            except Exception as e:
                print(f"  ✗ {spec_name}: {e}")
                continue
                
        self.robustness_results['specification_sensitivity'] = spec_results
        print(f"✓ Specification sensitivity: {len(spec_results)} specifications tested")
        
    def transformation_sensitivity(self):
        """Test sensitivity to different variable transformations"""
        
        print("\\nProgress: 92% | Testing transformation sensitivity...")
        
        # Create alternative transformations
        alt_data = self.data.copy()
        
        # Alternative inflation measures
        alt_data['inflation_qoq_annualized'] = alt_data['CPIAUCSL'].pct_change(3) * 400
        alt_data['inflation_mom_annualized'] = alt_data['CPIAUCSL'].pct_change(1) * 1200
        
        # Alternative unemployment gap measures
        alt_data['unemployment_level'] = alt_data['UNRATE']
        alt_data['unemployment_change'] = alt_data['UNRATE'].diff()
        
        # Alternative expectation measures
        alt_data['expectations_change'] = alt_data['MICH'].diff()
        alt_data['expectations_level'] = alt_data['MICH']
        
        transformation_tests = {
            'baseline_yoy': {
                'dependent': 'inflation',
                'independent': ['unemployment_gap', 'inflation_expectations']
            },
            'inflation_qoq': {
                'dependent': 'inflation_qoq_annualized',
                'independent': ['unemployment_gap', 'inflation_expectations']
            },
            'unemployment_level': {
                'dependent': 'inflation',
                'independent': ['unemployment_level', 'inflation_expectations']
            },
            'expectations_change': {
                'dependent': 'inflation',
                'independent': ['unemployment_gap', 'expectations_change']
            }
        }
        
        transformation_results = {}
        
        for test_name, spec in transformation_tests.items():
            
            try:
                dep_var = spec['dependent']
                indep_vars = spec['independent']
                
                # Check variable availability
                if dep_var not in alt_data.columns:
                    continue
                if not all(var in alt_data.columns for var in indep_vars):
                    continue
                    
                y = alt_data[dep_var].dropna()
                X = alt_data[indep_vars].dropna()
                
                common_idx = y.index.intersection(X.index)
                
                if len(common_idx) >= 30:
                    y_aligned = y.loc[common_idx]
                    X_aligned = sm.add_constant(X.loc[common_idx])
                    
                    model = sm.OLS(y_aligned, X_aligned).fit()
                    
                    transformation_results[test_name] = {
                        'dependent_var': dep_var,
                        'independent_vars': indep_vars,
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'observations': len(common_idx)
                    }
                    
                    print(f"  {test_name}: R² = {model.rsquared:.3f}")
                    
            except Exception as e:
                print(f"  ✗ {test_name}: {e}")
                continue
                
        self.robustness_results['transformation_sensitivity'] = transformation_results
        print(f"✓ Transformation sensitivity: {len(transformation_results)} transformations tested")
        
    def create_robustness_visualizations(self):
        """Create comprehensive robustness analysis visualizations"""
        
        print("\\nProgress: 95% | Creating robustness visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sample period sensitivity
        if 'sample_sensitivity' in self.robustness_results:
            sample_data = self.robustness_results['sample_sensitivity']
            
            periods = list(sample_data.keys())
            baseline_r2 = [sample_data[p]['baseline_r2'] for p in periods]
            enhanced_r2 = [sample_data[p]['enhanced_r2'] for p in periods]
            
            x = np.arange(len(periods))
            width = 0.35
            
            axes[0,0].bar(x - width/2, baseline_r2, width, label='Baseline', alpha=0.7, color='red')
            axes[0,0].bar(x + width/2, enhanced_r2, width, label='Enhanced', alpha=0.7, color='green')
            axes[0,0].set_xlabel('Sample Period')
            axes[0,0].set_ylabel('R-squared')
            axes[0,0].set_title('Sample Period Sensitivity')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(periods, rotation=45)
            axes[0,0].legend()
            
        # 2. Specification sensitivity
        if 'specification_sensitivity' in self.robustness_results:
            spec_data = self.robustness_results['specification_sensitivity']
            
            spec_names = list(spec_data.keys())
            r_squared_values = [spec_data[s]['r_squared'] for s in spec_names]
            
            axes[0,1].bar(spec_names, r_squared_values, alpha=0.7, color='skyblue')
            axes[0,1].set_xlabel('Model Specification')
            axes[0,1].set_ylabel('R-squared')
            axes[0,1].set_title('Specification Sensitivity')
            axes[0,1].tick_params(axis='x', rotation=45)
            
        # 3. AIC comparison across specifications
        if 'specification_sensitivity' in self.robustness_results:
            spec_data = self.robustness_results['specification_sensitivity']
            
            spec_names = list(spec_data.keys())
            aic_values = [spec_data[s]['aic'] for s in spec_names]
            
            axes[1,0].bar(spec_names, aic_values, alpha=0.7, color='orange')
            axes[1,0].set_xlabel('Model Specification')
            axes[1,0].set_ylabel('AIC (lower is better)')
            axes[1,0].set_title('AIC Comparison Across Specifications')
            axes[1,0].tick_params(axis='x', rotation=45)
            
        # 4. Transformation sensitivity
        if 'transformation_sensitivity' in self.robustness_results:
            transform_data = self.robustness_results['transformation_sensitivity']
            
            transform_names = list(transform_data.keys())
            transform_r2 = [transform_data[t]['r_squared'] for t in transform_names]
            
            axes[1,1].bar(transform_names, transform_r2, alpha=0.7, color='purple')
            axes[1,1].set_xlabel('Transformation')
            axes[1,1].set_ylabel('R-squared')
            axes[1,1].set_title('Transformation Sensitivity')
            axes[1,1].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig('charts/robustness_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Robustness visualizations created")
        
    def save_robustness_results(self):
        """Save all robustness analysis results"""
        
        # Save detailed results
        import json
        with open('outputs/robustness_analysis_results.json', 'w') as f:
            json.dump(self.robustness_results, f, indent=2, default=str)
            
        # Create summary report
        with open('outputs/robustness_analysis_report.txt', 'w') as f:
            f.write("ROBUSTNESS & SENSITIVITY ANALYSIS REPORT\\n")
            f.write("="*50 + "\\n\\n")
            
            # Sample sensitivity summary
            if 'sample_sensitivity' in self.robustness_results:
                sample_data = self.robustness_results['sample_sensitivity']
                f.write("SAMPLE PERIOD SENSITIVITY:\\n")
                f.write("-" * 30 + "\\n")
                
                for period, results in sample_data.items():
                    improvement = results['improvement']
                    f.write(f"{period:<15}: R² improvement = {improvement:+.3f}\\n")
                    
                f.write("\\n")
                
            # Specification sensitivity summary
            if 'specification_sensitivity' in self.robustness_results:
                spec_data = self.robustness_results['specification_sensitivity']
                f.write("SPECIFICATION SENSITIVITY:\\n")
                f.write("-" * 30 + "\\n")
                
                # Sort by R-squared
                sorted_specs = sorted(spec_data.items(), key=lambda x: x[1]['r_squared'], reverse=True)
                
                for spec_name, results in sorted_specs:
                    f.write(f"{spec_name:<20}: R² = {results['r_squared']:.3f}, AIC = {results['aic']:.1f}\\n")
                    
                f.write("\\n")
                
            # Overall assessment
            f.write("ROBUSTNESS ASSESSMENT:\\n")
            f.write("-" * 25 + "\\n")
            
            total_tests = sum(len(category) for category in self.robustness_results.values())
            f.write(f"Total robustness tests conducted: {total_tests}\\n")
            
            # Identify most robust specification
            if 'specification_sensitivity' in self.robustness_results:
                spec_data = self.robustness_results['specification_sensitivity']
                best_spec = max(spec_data.items(), key=lambda x: x[1]['r_squared'])
                f.write(f"Best performing specification: {best_spec[0]} (R² = {best_spec[1]['r_squared']:.3f})\\n")
                
        print("✓ Robustness analysis results saved")
        
        # Print summary
        print("\\n" + "="*70)
        print("ROBUSTNESS & SENSITIVITY ANALYSIS COMPLETE:")
        print("="*70)
        
        total_tests = sum(len(category) for category in self.robustness_results.values())
        print(f"Total robustness tests: {total_tests}")
        
        for category, results in self.robustness_results.items():
            print(f"  {category}: {len(results)} tests")
            
        print("✓ Model robustness comprehensively assessed")
        print("✓ Sensitivity to key assumptions evaluated")

if __name__ == "__main__":
    
    analyzer = RobustnessAnalyzer()
    
    # Execute comprehensive robustness analysis
    analyzer.load_comprehensive_data()
    analyzer.sample_period_sensitivity()
    analyzer.variable_specification_sensitivity()
    analyzer.transformation_sensitivity()
    analyzer.create_robustness_visualizations()
    analyzer.save_robustness_results()
    
    print("\\n✓ CRITICAL REQUIREMENT #6 COMPLETE: ROBUSTNESS ANALYSIS")
    print("Progress: 100% | Estimated remaining time: 1-2 hours")