"""
Phillips Curve Baseline Model - Undismal Protocol Step 2
Sparse baseline with defensible variables only
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# FRED API setup
fred = Fred(api_key='7a74bbd246e54e7458184b0137db2311')

class PhillipsCurveBaseline:
    def __init__(self):
        self.data = None
        self.baseline_model = None
        self.residuals = None
        
    def fetch_data(self, start_date='1990-01-01', end_date='2023-12-31'):
        """Fetch core Phillips Curve data from FRED"""
        
        print("Fetching baseline Phillips Curve data...")
        
        # Core Phillips Curve variables (defensible only)
        series = {
            'CPIAUCSL': 'Core CPI (Inflation)',           # Dependent variable
            'UNRATE': 'Unemployment Rate',                # Gap variable  
            'NROU': 'Natural Rate of Unemployment',       # For unemployment gap
            'MICH': 'Michigan Inflation Expectations'      # Forward-looking expectations
        }
        
        data_dict = {}
        for code, desc in series.items():
            try:
                data_dict[code] = fred.get_series(code, start_date, end_date)
                print(f"✓ {desc}")
            except Exception as e:
                print(f"✗ Failed to fetch {desc}: {e}")
                
        self.data = pd.DataFrame(data_dict)
        return self.data
    
    def prepare_variables(self):
        """Create model-ready variables"""
        
        # Calculate year-over-year inflation
        self.data['inflation'] = self.data['CPIAUCSL'].pct_change(12) * 100
        
        # Calculate unemployment gap
        self.data['unemployment_gap'] = self.data['UNRATE'] - self.data['NROU']
        
        # Use expectations directly (already in percent)
        self.data['inflation_expectations'] = self.data['MICH']
        
        # Drop missing values
        self.data = self.data.dropna()
        
        print(f"Data prepared: {len(self.data)} observations")
        print(f"Period: {self.data.index[0]} to {self.data.index[-1]}")
        
    def fit_baseline_model(self):
        """Fit sparse baseline Phillips Curve"""
        
        # Defensible specification: π = α + β₁(u-u*) + β₂πᵉ + ε
        y = self.data['inflation']
        X = self.data[['unemployment_gap', 'inflation_expectations']]
        X = sm.add_constant(X)
        
        self.baseline_model = sm.OLS(y, X).fit()
        self.residuals = self.baseline_model.resid
        
        return self.baseline_model
    
    def generate_diagnostics(self):
        """Generate comprehensive baseline diagnostics"""
        
        # Model summary
        print("\n" + "="*60)
        print("PHILLIPS CURVE BASELINE MODEL")
        print("="*60)
        print(self.baseline_model.summary())
        
        # Residual diagnostics
        print("\n" + "="*40)
        print("RESIDUAL DIAGNOSTICS")
        print("="*40)
        
        # Durbin-Watson test
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(self.residuals)
        print(f"Durbin-Watson statistic: {dw:.3f}")
        
        # Ljung-Box test for serial correlation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(self.residuals, lags=12, return_df=True)
        print(f"Ljung-Box p-value (lag 12): {lb_test['lb_pvalue'].iloc[-1]:.3f}")
        
        # Save results
        self.save_baseline_results()
        
    def save_baseline_results(self):
        """Save baseline model results and diagnostics"""
        
        # Model statistics
        results = {
            'R_squared': self.baseline_model.rsquared,
            'Adj_R_squared': self.baseline_model.rsquared_adj,
            'AIC': self.baseline_model.aic,
            'BIC': self.baseline_model.bic,
            'RMSE': np.sqrt(mean_squared_error(self.data['inflation'], 
                                             self.baseline_model.fittedvalues)),
            'N_observations': self.baseline_model.nobs
        }
        
        # Save to file
        results_df = pd.DataFrame([results])
        results_df.to_csv('outputs/baseline_model_stats.csv', index=False)
        
        # Save residuals for next step
        residuals_df = pd.DataFrame({
            'date': self.data.index,
            'residuals': self.residuals,
            'fitted_values': self.baseline_model.fittedvalues,
            'actual_inflation': self.data['inflation']
        })
        residuals_df.to_csv('outputs/baseline_residuals.csv', index=False)
        
        print("✓ Baseline results saved to outputs/")
    
    def plot_diagnostics(self):
        """Create diagnostic plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Fitted
        axes[0,0].scatter(self.baseline_model.fittedvalues, self.data['inflation'], alpha=0.6)
        axes[0,0].plot([self.data['inflation'].min(), self.data['inflation'].max()], 
                      [self.data['inflation'].min(), self.data['inflation'].max()], 'r--')
        axes[0,0].set_xlabel('Fitted Values')
        axes[0,0].set_ylabel('Actual Inflation')
        axes[0,0].set_title('Actual vs Fitted')
        
        # Residuals over time
        axes[0,1].plot(self.data.index, self.residuals)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals Over Time')
        
        # Residual histogram
        axes[1,0].hist(self.residuals, bins=30, alpha=0.7)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Residual Distribution')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig('charts/baseline_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Diagnostic plots saved to charts/baseline_diagnostics.png")

if __name__ == "__main__":
    # Execute baseline model
    pc = PhillipsCurveBaseline()
    
    print("UNDISMAL PROTOCOL - STEP 2: SPARSE BASELINE")
    print("=" * 50)
    
    # Fetch and prepare data
    pc.fetch_data()
    pc.prepare_variables()
    
    # Fit baseline model
    model = pc.fit_baseline_model()
    
    # Generate diagnostics
    pc.generate_diagnostics()
    pc.plot_diagnostics()
    
    print("\n✓ STEP 2 COMPLETE: Sparse baseline established")
    print(f"R²: {model.rsquared:.3f}")
    print(f"Observations: {model.nobs}")
    print("Next: Residual analysis for variable candidates")