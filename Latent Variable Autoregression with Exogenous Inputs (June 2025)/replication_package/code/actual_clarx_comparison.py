#!/usr/bin/env python3
"""
ACTUAL CLARX PERFORMANCE COMPARISON
Author: Matthew Busigin (matt@voxgenius.ai)
Date: July 28, 2025

This script implements ACTUAL performance comparison between original and improved CLARX
methodologies using REAL data and REAL model implementations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ActualCLARXComparison:
    """Implement actual CLARX comparison with real performance metrics"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_real_data(self):
        """Load the actual merged economic/financial data"""
        print("Loading real economic and financial data...")
        self.data = pd.read_csv('../data/merged_data.csv', index_col=0, parse_dates=True)
        
        # Remove any rows with NaN values
        self.data = self.data.dropna()
        print(f"Loaded {len(self.data)} observations with {self.data.shape[1]} variables")
        
    def prepare_features_targets(self, horizon=1):
        """Prepare features (X) and targets (y) for forecasting"""
        # Use lagged values as features
        X = self.data.iloc[:-horizon].values
        y = self.data['GDP_growth'].iloc[horizon:].values
        
        # Ensure same length
        X = X[:len(y)]
        
        return X, y
        
    def original_clarx(self, X, y, n_latent=3, max_iter=100, tol=1e-4):
        """
        Implement original CLARX as described in Bargman (2025)
        Note: This is a simplified version focusing on key issues
        """
        n_samples, n_features = X.shape
        
        # Initialize randomly (as in original)
        W = np.random.randn(n_features, n_latent)
        W = W / np.linalg.norm(W, axis=0)
        phi = np.random.randn(n_latent)
        
        converged = False
        for iteration in range(max_iter):
            # Extract latent variables
            L = X @ W
            
            # Update autoregressive parameters (simplified)
            phi_new = np.linalg.lstsq(L[:-1], y[1:], rcond=None)[0]
            
            # Update projection matrix (simplified version of original)
            residuals = y[1:] - L[:-1] @ phi_new
            W_new = np.linalg.lstsq(X[:-1].T @ X[:-1], X[:-1].T @ residuals, rcond=None)[0]
            
            # Check convergence (often fails)
            if np.linalg.norm(W_new - W) < tol and np.linalg.norm(phi_new - phi) < tol:
                converged = True
                break
                
            W = W_new
            phi = phi_new
            
        return W, phi, converged, iteration + 1
        
    def improved_clarx(self, X, y, n_latent=3, max_iter=100, tol=1e-4):
        """
        Implement improved CLARX with stability enhancements
        """
        n_samples, n_features = X.shape
        
        # Better initialization using PCA
        pca = PCA(n_components=n_latent)
        L_init = pca.fit_transform(X)
        W = pca.components_.T
        
        # Initialize phi using regularized regression
        ridge = Ridge(alpha=0.1)
        ridge.fit(L_init[:-1], y[1:])
        phi = ridge.coef_
        
        # Adaptive learning rate
        alpha = 0.5
        
        converged = False
        for iteration in range(max_iter):
            # Extract latent variables with current W
            L = X @ W
            
            # Update phi with regularization
            ridge.fit(L[:-1], y[1:])
            phi_new = ridge.coef_
            
            # Update W with stability constraints
            grad = -2 * X[:-1].T @ (y[1:] - L[:-1] @ phi_new).reshape(-1, 1) @ phi_new.reshape(1, -1)
            W_new = W - alpha * grad / (n_samples - 1)
            
            # Normalize columns
            W_new = W_new / np.linalg.norm(W_new, axis=0)
            
            # Check convergence
            if np.linalg.norm(W_new - W) < tol and np.linalg.norm(phi_new - phi) < tol:
                converged = True
                break
                
            # Adaptive learning rate
            if iteration > 10:
                alpha *= 0.95
                
            W = W_new
            phi = phi_new
            
        return W, phi, converged, iteration + 1
        
    def evaluate_forecasting_performance(self):
        """Run actual forecasting comparison on real data"""
        print("\nRunning actual forecasting performance comparison...")
        
        # Prepare data
        X, y = self.prepare_features_targets()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        methods = {
            'Original_CLARX': self.original_clarx,
            'Improved_CLARX': self.improved_clarx,
            'Ridge_Baseline': lambda X, y, **kwargs: self._ridge_baseline(X, y),
            'PCA_Regression': lambda X, y, **kwargs: self._pca_baseline(X, y)
        }
        
        results = {method: {
            'rmse': [], 'mae': [], 'directional': [], 
            'converged': [], 'iterations': []
        } for method in methods}
        
        # Run cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            print(f"\nProcessing fold {fold + 1}/5...")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for method_name, method_func in methods.items():
                try:
                    if 'CLARX' in method_name:
                        W, phi, converged, iterations = method_func(X_train, y_train)
                        
                        # Make predictions
                        L_test = X_test @ W
                        y_pred = L_test @ phi
                        
                        results[method_name]['converged'].append(converged)
                        results[method_name]['iterations'].append(iterations)
                    else:
                        # Baseline methods return predictions directly
                        y_pred, converged, iterations = method_func(X_train, y_train)
                        y_pred = y_pred[test_idx]
                        results[method_name]['converged'].append(True)
                        results[method_name]['iterations'].append(1)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Directional accuracy
                    if len(y_test) > 1:
                        direction_actual = np.sign(np.diff(y_test))
                        direction_pred = np.sign(np.diff(y_pred))
                        directional = np.mean(direction_actual == direction_pred)
                    else:
                        directional = 0.5
                    
                    results[method_name]['rmse'].append(rmse)
                    results[method_name]['mae'].append(mae)
                    results[method_name]['directional'].append(directional)
                    
                except Exception as e:
                    print(f"Error in {method_name}: {e}")
                    # Record failure
                    results[method_name]['rmse'].append(np.nan)
                    results[method_name]['mae'].append(np.nan)
                    results[method_name]['directional'].append(np.nan)
                    results[method_name]['converged'].append(False)
                    results[method_name]['iterations'].append(100)
        
        # Summarize results
        summary = {}
        for method, metrics in results.items():
            summary[method] = {
                'RMSE_mean': np.nanmean(metrics['rmse']),
                'RMSE_std': np.nanstd(metrics['rmse']),
                'MAE_mean': np.nanmean(metrics['mae']),
                'MAE_std': np.nanstd(metrics['mae']),
                'Directional_mean': np.nanmean(metrics['directional']),
                'Directional_std': np.nanstd(metrics['directional']),
                'Convergence_rate': np.mean(metrics['converged']),
                'Avg_iterations': np.mean(metrics['iterations'])
            }
        
        self.results = summary
        return summary
        
    def _ridge_baseline(self, X_train, y_train):
        """Ridge regression baseline"""
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train[:-1], y_train[1:])
        y_pred = ridge.predict(X_train)
        return y_pred, True, 1
        
    def _pca_baseline(self, X_train, y_train):
        """PCA + regression baseline"""
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_train)
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_pca[:-1], y_train[1:])
        y_pred = ridge.predict(X_pca)
        return y_pred, True, 1
        
    def print_results(self):
        """Print actual performance comparison results"""
        print("\n" + "="*80)
        print("ACTUAL PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        for method, metrics in self.results.items():
            print(f"\n{method}:")
            print(f"  RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}")
            print(f"  MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}")
            print(f"  Directional Accuracy: {metrics['Directional_mean']:.3f} ± {metrics['Directional_std']:.3f}")
            print(f"  Convergence Rate: {metrics['Convergence_rate']*100:.1f}%")
            print(f"  Average Iterations: {metrics['Avg_iterations']:.1f}")
        
        # Calculate actual improvements
        orig_rmse = self.results['Original_CLARX']['RMSE_mean']
        imp_rmse = self.results['Improved_CLARX']['RMSE_mean']
        actual_improvement = (orig_rmse - imp_rmse) / orig_rmse * 100
        
        print(f"\nACTUAL IMPROVEMENT: {actual_improvement:.1f}% reduction in RMSE")
        
    def save_results(self):
        """Save actual results to CSV"""
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('../data/actual_performance_comparison.csv')
        print("\nResults saved to actual_performance_comparison.csv")


if __name__ == "__main__":
    # Run actual comparison
    comparison = ActualCLARXComparison()
    comparison.load_real_data()
    comparison.evaluate_forecasting_performance()
    comparison.print_results()
    comparison.save_results()