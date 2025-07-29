"""
Phillips Curve Residual Analysis - Undismal Protocol Step 3
Let residuals issue work orders: Diagnose structure → hypotheses
Agent: Leibniz, VoxGenius Inc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

class ResidualAnalyzer:
    def __init__(self):
        self.residuals_data = None
        self.work_orders = []
        
    def load_baseline_residuals(self):
        """Load residuals from baseline model"""
        self.residuals_data = pd.read_csv('outputs/baseline_residuals.csv')
        self.residuals_data['date'] = pd.to_datetime(self.residuals_data['date'])
        print(f"✓ Loaded {len(self.residuals_data)} residual observations")
        
    def diagnose_temporal_structure(self):
        """Diagnose temporal patterns in residuals"""
        
        residuals = self.residuals_data['residuals'].values
        
        # Autocorrelation analysis
        print("\n" + "="*50)
        print("TEMPORAL STRUCTURE DIAGNOSIS")
        print("="*50)
        
        # Calculate ACF and PACF
        acf_vals = acf(residuals, nlags=20, alpha=0.05)
        pacf_vals = pacf(residuals, nlags=20, alpha=0.05)
        
        # Identify significant lags
        sig_acf_lags = []
        for i, (val, conf) in enumerate(zip(acf_vals[0][1:], acf_vals[1][1:]), 1):
            if val < conf[0] or val > conf[1]:
                sig_acf_lags.append(i)
                
        print(f"Significant ACF lags: {sig_acf_lags[:10]}")  # First 10
        
        # Issue work orders based on patterns
        if len(sig_acf_lags) > 5:
            self.work_orders.append({
                'type': 'PERSISTENT_AUTOCORRELATION',
                'priority': 'HIGH',
                'description': 'Strong serial correlation suggests missing dynamic variables',
                'candidates': ['lagged_inflation', 'monetary_policy_vars', 'persistence_factors']
            })
            
        # Seasonal patterns
        seasonal_lags = [i for i in sig_acf_lags if i in [3, 6, 12]]
        if seasonal_lags:
            self.work_orders.append({
                'type': 'SEASONAL_PATTERNS',
                'priority': 'MEDIUM', 
                'description': f'Seasonal autocorrelation at lags {seasonal_lags}',
                'candidates': ['seasonal_dummies', 'weather_vars', 'commodity_cycles']
            })
            
    def diagnose_volatility_structure(self):
        """Diagnose heteroscedasticity and volatility patterns"""
        
        residuals = self.residuals_data['residuals'].values
        dates = self.residuals_data['date']
        
        print("\n" + "="*50)
        print("VOLATILITY STRUCTURE DIAGNOSIS") 
        print("="*50)
        
        # Rolling standard deviation
        window = 12
        rolling_std = pd.Series(residuals).rolling(window).std()
        
        # Identify high volatility periods
        vol_threshold = rolling_std.quantile(0.8)
        high_vol_periods = dates[rolling_std > vol_threshold]
        
        print(f"High volatility periods identified: {len(high_vol_periods)}")
        
        # Check for ARCH effects
        residuals_sq = residuals**2
        arch_acf = acf(residuals_sq[window:], nlags=12, alpha=0.05)
        
        sig_arch_lags = []
        for i, (val, conf) in enumerate(zip(arch_acf[0][1:], arch_acf[1][1:]), 1):
            if val < conf[0] or val > conf[1]:
                sig_arch_lags.append(i)
                
        if len(sig_arch_lags) > 2:
            self.work_orders.append({
                'type': 'VOLATILITY_CLUSTERING',
                'priority': 'MEDIUM',
                'description': 'ARCH effects suggest time-varying volatility',
                'candidates': ['uncertainty_indices', 'financial_stress', 'regime_variables']
            })
            
    def diagnose_structural_breaks(self):
        """Identify potential structural breaks"""
        
        residuals = self.residuals_data['residuals'].values
        dates = self.residuals_data['date']
        
        print("\n" + "="*50)
        print("STRUCTURAL BREAK DIAGNOSIS")
        print("="*50)
        
        # Simple break test: Large residual periods
        residual_abs = np.abs(residuals)
        break_threshold = np.percentile(residual_abs, 95)
        
        large_residual_mask = residual_abs > break_threshold
        large_residual_periods = dates[large_residual_mask]
        
        # Group consecutive periods
        break_periods = []
        if len(large_residual_periods) > 0:
            current_period = [large_residual_periods.iloc[0]]
            
            for date in large_residual_periods.iloc[1:]:
                if (date - current_period[-1]).days <= 90:  # Within 3 months
                    current_period.append(date)
                else:
                    if len(current_period) >= 2:
                        break_periods.append(current_period)
                    current_period = [date]
                    
            if len(current_period) >= 2:
                break_periods.append(current_period)
        
        print(f"Potential structural break periods: {len(break_periods)}")
        for i, period in enumerate(break_periods[:5]):  # Show first 5
            print(f"  Break {i+1}: {period[0].strftime('%Y-%m')} to {period[-1].strftime('%Y-%m')}")
            
        if len(break_periods) > 0:
            self.work_orders.append({
                'type': 'STRUCTURAL_BREAKS',
                'priority': 'HIGH',
                'description': f'Identified {len(break_periods)} potential break periods',
                'candidates': ['crisis_dummies', 'regime_switches', 'policy_change_vars']
            })
            
    def diagnose_distributional_properties(self):
        """Analyze distributional properties of residuals"""
        
        residuals = self.residuals_data['residuals'].values
        
        print("\n" + "="*50)
        print("DISTRIBUTIONAL DIAGNOSIS")
        print("="*50)
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        
        print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
        print(f"Jarque-Bera p-value: {jb_p:.4f}")
        
        # Skewness and kurtosis
        skewness = stats.skew(residuals)
        kurt = stats.kurtosis(residuals)
        
        print(f"Skewness: {skewness:.3f}")
        print(f"Excess Kurtosis: {kurt:.3f}")
        
        if abs(skewness) > 0.5:
            self.work_orders.append({
                'type': 'ASYMMETRIC_SHOCKS',
                'priority': 'MEDIUM',
                'description': f'High skewness ({skewness:.2f}) suggests asymmetric responses',
                'candidates': ['asymmetric_policy_vars', 'business_cycle_indicators', 'threshold_models']
            })
            
        if kurt > 3:
            self.work_orders.append({
                'type': 'FAT_TAILS', 
                'priority': 'MEDIUM',
                'description': f'Excess kurtosis ({kurt:.2f}) suggests fat-tailed innovations',
                'candidates': ['extreme_event_vars', 'uncertainty_measures', 'jump_components']
            })
            
    def generate_hypotheses(self):
        """Generate economic hypotheses from residual patterns"""
        
        print("\n" + "="*60)
        print("ECONOMIC HYPOTHESES FROM RESIDUAL ANALYSIS")
        print("="*60)
        
        hypothesis_map = {
            'PERSISTENT_AUTOCORRELATION': [
                "Inflation has intrinsic persistence not captured by expectations",
                "Monetary policy transmission operates with longer lags",
                "Supply-side factors create persistent inflation pressures"
            ],
            'SEASONAL_PATTERNS': [
                "Energy price seasonality affects core inflation",
                "Labor market dynamics have seasonal components",
                "Fiscal policy has predictable timing effects"
            ],
            'VOLATILITY_CLUSTERING': [
                "Financial market stress spills over to inflation",
                "Policy uncertainty creates time-varying inflation volatility",
                "External shocks cluster in time"
            ],
            'STRUCTURAL_BREAKS': [
                "Monetary policy regime changes alter Phillips Curve",
                "Globalization changed inflation dynamics",
                "Financial crises create persistent shifts"
            ],
            'ASYMMETRIC_SHOCKS': [
                "Positive and negative demand shocks have different effects",
                "Policy responses are asymmetric over business cycle",
                "Supply disruptions have non-linear effects"
            ],
            'FAT_TAILS': [
                "Extreme events drive inflation more than gradual changes",
                "Policy surprises have outsized impacts",
                "Tail risks from external sector are important"
            ]
        }
        
        hypotheses = []
        for order in self.work_orders:
            order_hypotheses = hypothesis_map.get(order['type'], [])
            for hypothesis in order_hypotheses:
                hypotheses.append({
                    'hypothesis': hypothesis,
                    'priority': order['priority'],
                    'evidence': order['description'],
                    'variable_candidates': order['candidates']
                })
                
        # Save hypotheses
        hypotheses_df = pd.DataFrame(hypotheses)
        hypotheses_df.to_csv('outputs/economic_hypotheses.csv', index=False)
        
        print(f"Generated {len(hypotheses)} economic hypotheses")
        for i, h in enumerate(hypotheses[:5]):  # Show first 5
            print(f"\n{i+1}. {h['hypothesis']}")
            print(f"   Evidence: {h['evidence']}")
            print(f"   Candidates: {', '.join(h['variable_candidates'][:3])}")
            
    def create_diagnostic_plots(self):
        """Create comprehensive residual diagnostic plots"""
        
        residuals = self.residuals_data['residuals'].values
        dates = self.residuals_data['date']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Time series plot
        axes[0,0].plot(dates, residuals, alpha=0.7)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_title('Residuals Over Time')
        axes[0,0].set_ylabel('Residuals')
        
        # Autocorrelation function
        acf_vals = acf(residuals, nlags=20, alpha=0.05)
        lags = range(len(acf_vals[0]))
        axes[0,1].stem(lags, acf_vals[0])
        axes[0,1].fill_between(lags, acf_vals[1][:,0], acf_vals[1][:,1], alpha=0.2)
        axes[0,1].set_title('Autocorrelation Function')
        axes[0,1].set_xlabel('Lag')
        
        # Rolling volatility
        rolling_std = pd.Series(residuals).rolling(12).std()
        axes[1,0].plot(dates, rolling_std)
        axes[1,0].set_title('Rolling Standard Deviation (12-month)')
        axes[1,0].set_ylabel('Volatility')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Normal Q-Q Plot')
        
        # Histogram with normal overlay
        axes[2,0].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[2,0].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                      'r-', linewidth=2, label='Normal')
        axes[2,0].set_title('Residual Distribution')
        axes[2,0].legend()
        
        # Squared residuals (ARCH test)
        residuals_sq = residuals**2
        axes[2,1].plot(dates, residuals_sq, alpha=0.7)
        axes[2,1].set_title('Squared Residuals (ARCH Test)')
        axes[2,1].set_ylabel('Residuals²')
        
        plt.tight_layout()
        plt.savefig('charts/residual_work_orders.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_work_orders(self):
        """Save work orders to file"""
        
        work_orders_df = pd.DataFrame(self.work_orders)
        work_orders_df.to_csv('outputs/residual_work_orders.csv', index=False)
        
        print(f"\n✓ Saved {len(self.work_orders)} work orders to outputs/")
        
        # Summary report
        print("\n" + "="*60)
        print("WORK ORDERS SUMMARY")
        print("="*60)
        
        for i, order in enumerate(self.work_orders, 1):
            print(f"\n{i}. {order['type']} ({order['priority']} PRIORITY)")
            print(f"   {order['description']}")
            print(f"   Candidates: {', '.join(order['candidates'])}")

if __name__ == "__main__":
    print("UNDISMAL PROTOCOL - STEP 3: RESIDUAL WORK ORDERS")
    print("=" * 60)
    
    analyzer = ResidualAnalyzer()
    
    # Load baseline residuals
    analyzer.load_baseline_residuals()
    
    # Comprehensive residual diagnosis
    analyzer.diagnose_temporal_structure()
    analyzer.diagnose_volatility_structure() 
    analyzer.diagnose_structural_breaks()
    analyzer.diagnose_distributional_properties()
    
    # Generate economic hypotheses
    analyzer.generate_hypotheses()
    
    # Create diagnostic plots
    analyzer.create_diagnostic_plots()
    
    # Save results
    analyzer.save_work_orders()
    
    print("\n✓ STEP 3 COMPLETE: Residual work orders issued")
    print("Next: Theory-scoped candidate assembly")