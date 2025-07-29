"""
Complete Enhanced Model Implementation - Critical Requirement #2
Fit actual enhanced model with comprehensive diagnostics
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# FRED API setup
fred = Fred(api_key='7a74bbd246e54e7458184b0137db2311')

class CompleteEnhancedModel:
    def __init__(self):
        self.data = None
        self.baseline_model = None
        self.enhanced_model = None
        self.diagnostics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare complete dataset"""
        
        print("CRITICAL REQUIREMENT #2: COMPLETE ENHANCED MODEL")
        print("=" * 55)
        print("Progress: 17% | Loading and preparing data...")
        
        # Load the OOS validation results to inform our model specification
        try:
            oos_results = pd.read_csv('outputs/oos_performance_summary.csv', index_col=0)
            best_model = oos_results.loc[oos_results['RMSE'].idxmin()]
            print(f"✓ Best OOS model identified: {best_model.name} (RMSE: {best_model['RMSE']:.4f})")
        except:
            print("⚠ Using default enhanced specification")
            
        # Fetch comprehensive data
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
                
        self.data = pd.DataFrame(data_dict)
        
        # Create all model variables
        self.create_model_variables()
        print(f"✓ Dataset prepared: {len(self.data)} observations")
        
    def create_model_variables(self):
        """Create comprehensive set of model variables"""
        
        # Dependent variable
        self.data['inflation'] = self.data['CPIAUCSL'].pct_change(12) * 100
        
        # Baseline Phillips Curve variables
        self.data['unemployment_gap'] = self.data['UNRATE'] - self.data['NROU']
        self.data['inflation_expectations'] = self.data['MICH']
        
        # Enhanced variables (based on OOS validation results)
        self.data['dollar_yoy'] = self.data['DTWEXBGS'].pct_change(12) * 100
        self.data['dollar_yoy_lag12'] = self.data['dollar_yoy'].shift(12)
        
        self.data['breakeven_5y'] = self.data['T5YIE']
        self.data['breakeven_5y_lag3'] = self.data['breakeven_5y'].shift(3)
        
        self.data['financial_stress'] = self.data['STLFSI4']
        self.data['financial_stress_lag1'] = self.data['financial_stress'].shift(1)
        
        # Additional variables for robustness
        self.data['treasury_10y'] = self.data['GS10']
        self.data['weekly_hours_mfg'] = self.data['AWHMAN']
        
        # Clean data
        required_vars = ['inflation', 'unemployment_gap', 'inflation_expectations']
        self.data = self.data.dropna(subset=required_vars)
        
    def fit_baseline_model(self):
        """Fit baseline Phillips Curve model"""
        
        print("\\nProgress: 22% | Fitting baseline Phillips Curve model...")
        
        # Baseline specification
        y = self.data['inflation'].dropna()
        X = self.data[['unemployment_gap', 'inflation_expectations']].dropna()
        
        # Align data
        common_idx = y.index.intersection(X.index)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]
        X_aligned = sm.add_constant(X_aligned)
        
        # Fit model
        self.baseline_model = sm.OLS(y_aligned, X_aligned).fit()
        
        print(f"✓ Baseline model fitted")
        print(f"  R²: {self.baseline_model.rsquared:.4f}")
        print(f"  Observations: {self.baseline_model.nobs}")
        
    def fit_enhanced_model(self):
        """Fit complete enhanced Phillips Curve model"""
        
        print("\\nProgress: 30% | Fitting enhanced Phillips Curve model...")
        
        # Enhanced specification (based on OOS validation)
        enhanced_vars = [
            'unemployment_gap', 
            'inflation_expectations',
            'dollar_yoy_lag12',
            'breakeven_5y_lag3'
        ]
        
        # Prepare data
        y = self.data['inflation'].dropna()
        X = self.data[enhanced_vars].dropna()
        
        # Align data
        common_idx = y.index.intersection(X.index)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]
        X_aligned = sm.add_constant(X_aligned)
        
        # Fit enhanced model
        self.enhanced_model = sm.OLS(y_aligned, X_aligned).fit()
        
        print(f"✓ Enhanced model fitted")
        print(f"  R²: {self.enhanced_model.rsquared:.4f}")
        print(f"  Observations: {self.enhanced_model.nobs}")
        print(f"  Improvement: +{(self.enhanced_model.rsquared - self.baseline_model.rsquared):.4f}")
        
    def comprehensive_model_diagnostics(self):
        """Perform comprehensive model diagnostics"""
        
        print("\\nProgress: 45% | Conducting comprehensive diagnostics...")
        
        models = {
            'baseline': self.baseline_model,
            'enhanced': self.enhanced_model
        }
        
        for model_name, model in models.items():
            
            print(f"\\n{model_name.upper()} MODEL DIAGNOSTICS:")
            print("-" * 35)
            
            residuals = model.resid
            fitted = model.fittedvalues
            
            # 1. Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            jb_stat, jb_p = jarque_bera(residuals)
            
            # 2. Serial correlation
            dw_stat = durbin_watson(residuals)
            lb_test = acorr_ljungbox(residuals, lags=12, return_df=True)
            lb_p_min = lb_test['lb_pvalue'].min()
            
            # 3. Heteroscedasticity
            _, het_p, _, _ = het_white(residuals, model.model.exog)
            
            # 4. Multicollinearity (VIF)
            vif_data = pd.DataFrame()
            if model.model.exog.shape[1] > 1:  # More than just constant
                vif_data['Variable'] = model.model.exog_names
                vif_data['VIF'] = [variance_inflation_factor(model.model.exog, i) 
                                  for i in range(model.model.exog.shape[1])]
                max_vif = vif_data['VIF'][1:].max()  # Exclude constant
            else:
                max_vif = np.nan
                
            # Store diagnostics
            self.diagnostics[model_name] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'aic': model.aic,
                'bic': model.bic,
                'observations': model.nobs,
                'shapiro_p': shapiro_p,
                'jarque_bera_p': jb_p,
                'durbin_watson': dw_stat,
                'ljung_box_p_min': lb_p_min,
                'white_test_p': het_p,
                'max_vif': max_vif,
                'residual_std': residuals.std(),
                'residual_mean': residuals.mean()
            }
            
            # Print key diagnostics
            print(f"R²: {model.rsquared:.4f} | Adj R²: {model.rsquared_adj:.4f}")
            print(f"AIC: {model.aic:.2f} | BIC: {model.bic:.2f}")
            print(f"Durbin-Watson: {dw_stat:.3f}")
            print(f"Ljung-Box p-val: {lb_p_min:.4f}")
            print(f"White test p-val: {het_p:.4f}")
            if not np.isnan(max_vif):
                print(f"Max VIF: {max_vif:.2f}")
                
        print("\\n✓ Comprehensive diagnostics completed")
        
    def model_comparison_tests(self):
        """Perform formal model comparison tests"""
        
        print("\\nProgress: 60% | Conducting model comparison tests...")
        
        # F-test for nested models
        baseline_ssr = self.baseline_model.ssr
        enhanced_ssr = self.enhanced_model.ssr
        
        # Degrees of freedom
        df_baseline = self.baseline_model.df_resid
        df_enhanced = self.enhanced_model.df_resid
        df_diff = df_baseline - df_enhanced
        
        # F-statistic
        f_stat = ((baseline_ssr - enhanced_ssr) / df_diff) / (enhanced_ssr / df_enhanced)
        f_p_value = 1 - stats.f.cdf(f_stat, df_diff, df_enhanced)
        
        # Likelihood ratio test
        lr_stat = 2 * (self.enhanced_model.llf - self.baseline_model.llf)
        lr_p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
        
        # Information criteria comparison
        aic_improvement = self.baseline_model.aic - self.enhanced_model.aic
        bic_improvement = self.baseline_model.bic - self.enhanced_model.bic
        
        comparison_results = {
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'lr_statistic': lr_stat,
            'lr_p_value': lr_p_value,
            'aic_improvement': aic_improvement,
            'bic_improvement': bic_improvement,
            'r_squared_improvement': self.enhanced_model.rsquared - self.baseline_model.rsquared
        }
        
        self.diagnostics['comparison'] = comparison_results
        
        print(f"F-test p-value: {f_p_value:.6f}")
        print(f"LR test p-value: {lr_p_value:.6f}")
        print(f"AIC improvement: {aic_improvement:.2f}")
        print(f"BIC improvement: {bic_improvement:.2f}")
        print(f"R² improvement: {comparison_results['r_squared_improvement']:.4f}")
        
    def create_model_visualizations(self):
        """Create comprehensive model visualization"""
        
        print("\\nProgress: 75% | Creating model visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Model fit comparison
        y_actual = pd.Series(self.enhanced_model.model.endog, index=self.enhanced_model.fittedvalues.index)
        baseline_fitted = self.baseline_model.fittedvalues.reindex(y_actual.index)
        enhanced_fitted = self.enhanced_model.fittedvalues
        
        axes[0,0].scatter(baseline_fitted, y_actual, alpha=0.6, label='Baseline', color='red')
        axes[0,0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
        axes[0,0].set_xlabel('Fitted Values')
        axes[0,0].set_ylabel('Actual Inflation')
        axes[0,0].set_title('Baseline Model: Actual vs Fitted')
        axes[0,0].legend()
        
        axes[0,1].scatter(enhanced_fitted, y_actual, alpha=0.6, label='Enhanced', color='green')
        axes[0,1].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
        axes[0,1].set_xlabel('Fitted Values')
        axes[0,1].set_ylabel('Actual Inflation')
        axes[0,1].set_title('Enhanced Model: Actual vs Fitted')
        axes[0,1].legend()
        
        # 2. Residual plots
        baseline_resid = self.baseline_model.resid.reindex(y_actual.index)
        enhanced_resid = self.enhanced_model.resid
        
        axes[1,0].plot(baseline_resid.index, baseline_resid, alpha=0.7, color='red')
        axes[1,0].axhline(y=0, color='black', linestyle='--')
        axes[1,0].set_title('Baseline Model Residuals')
        axes[1,0].set_ylabel('Residuals')
        
        axes[1,1].plot(enhanced_resid.index, enhanced_resid, alpha=0.7, color='green')
        axes[1,1].axhline(y=0, color='black', linestyle='--')
        axes[1,1].set_title('Enhanced Model Residuals')
        axes[1,1].set_ylabel('Residuals')
        
        # 3. Residual distributions
        axes[2,0].hist(baseline_resid.dropna(), bins=20, alpha=0.7, color='red', label='Baseline')
        axes[2,0].hist(enhanced_resid, bins=20, alpha=0.7, color='green', label='Enhanced')
        axes[2,0].set_title('Residual Distributions')
        axes[2,0].set_xlabel('Residuals')
        axes[2,0].legend()
        
        # 4. Model performance comparison
        metrics = ['R²', 'Adj R²', 'AIC', 'BIC']
        baseline_metrics = [
            self.diagnostics['baseline']['r_squared'],
            self.diagnostics['baseline']['adj_r_squared'],
            self.diagnostics['baseline']['aic'],
            self.diagnostics['baseline']['bic']
        ]
        enhanced_metrics = [
            self.diagnostics['enhanced']['r_squared'],
            self.diagnostics['enhanced']['adj_r_squared'],
            self.diagnostics['enhanced']['aic'],
            self.diagnostics['enhanced']['bic']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize AIC and BIC for visualization (lower is better)
        baseline_metrics_viz = baseline_metrics.copy()
        enhanced_metrics_viz = enhanced_metrics.copy()
        for i in [2, 3]:  # AIC and BIC indices
            baseline_metrics_viz[i] = -baseline_metrics_viz[i]
            enhanced_metrics_viz[i] = -enhanced_metrics_viz[i]
        
        axes[2,1].bar(x - width/2, baseline_metrics_viz, width, label='Baseline', color='red', alpha=0.7)
        axes[2,1].bar(x + width/2, enhanced_metrics_viz, width, label='Enhanced', color='green', alpha=0.7)
        axes[2,1].set_xlabel('Metrics')
        axes[2,1].set_title('Model Performance Comparison')
        axes[2,1].set_xticks(x)
        axes[2,1].set_xticklabels(metrics)
        axes[2,1].legend()
        
        plt.tight_layout()
        plt.savefig('charts/complete_enhanced_model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Model visualizations created")
        
    def save_model_results(self):
        """Save complete model results and diagnostics"""
        
        print("\\nProgress: 90% | Saving model results...")
        
        # Save model summaries
        with open('outputs/baseline_model_summary.txt', 'w') as f:
            f.write("BASELINE PHILLIPS CURVE MODEL\\n")
            f.write("="*40 + "\\n\\n")
            f.write(str(self.baseline_model.summary()))
            
        with open('outputs/enhanced_model_summary.txt', 'w') as f:
            f.write("ENHANCED PHILLIPS CURVE MODEL\\n")
            f.write("="*40 + "\\n\\n")
            f.write(str(self.enhanced_model.summary()))
            
        # Save diagnostics
        diagnostics_df = pd.DataFrame(self.diagnostics).T
        diagnostics_df.to_csv('outputs/complete_model_diagnostics.csv')
        
        # Save detailed results
        results_summary = {
            'baseline_model': {
                'r_squared': float(self.baseline_model.rsquared),
                'adj_r_squared': float(self.baseline_model.rsquared_adj),
                'aic': float(self.baseline_model.aic),
                'bic': float(self.baseline_model.bic),
                'observations': int(self.baseline_model.nobs),
                'variables': self.baseline_model.model.exog_names
            },
            'enhanced_model': {
                'r_squared': float(self.enhanced_model.rsquared),
                'adj_r_squared': float(self.enhanced_model.rsquared_adj),
                'aic': float(self.enhanced_model.aic),
                'bic': float(self.enhanced_model.bic),
                'observations': int(self.enhanced_model.nobs),
                'variables': self.enhanced_model.model.exog_names
            },
            'improvement': {
                'r_squared_gain': float(self.enhanced_model.rsquared - self.baseline_model.rsquared),
                'aic_improvement': float(self.baseline_model.aic - self.enhanced_model.aic),
                'bic_improvement': float(self.baseline_model.bic - self.enhanced_model.bic)
            }
        }
        
        import json
        with open('outputs/complete_model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print("✓ Model results saved")
        
        # Print final summary
        print("\\n" + "="*55)
        print("MODEL FITTING COMPLETE - KEY RESULTS:")
        print("="*55)
        print(f"Baseline R²: {self.baseline_model.rsquared:.4f}")
        print(f"Enhanced R²: {self.enhanced_model.rsquared:.4f}")
        print(f"Improvement: +{(self.enhanced_model.rsquared - self.baseline_model.rsquared):.4f}")
        print(f"F-test p-value: {self.diagnostics['comparison']['f_p_value']:.6f}")

if __name__ == "__main__":
    
    modeler = CompleteEnhancedModel()
    
    # Execute complete model fitting
    modeler.load_and_prepare_data()
    modeler.fit_baseline_model()
    modeler.fit_enhanced_model()
    modeler.comprehensive_model_diagnostics()
    modeler.model_comparison_tests()
    modeler.create_model_visualizations()
    modeler.save_model_results()
    
    print("\\n✓ CRITICAL REQUIREMENT #2 COMPLETE: ENHANCED MODEL FITTED")
    print("Progress: 100% | Estimated remaining time: 5-6 hours")