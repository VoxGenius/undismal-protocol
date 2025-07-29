#!/usr/bin/env python3
"""
Runner script for real data analysis
Executes the enhanced economic forecasting with actual data sources
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/real_data_run.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Execute real data analysis"""
    
    print("="*60)
    print("ECONOMIC FORECASTING WITH REAL DATA SOURCES")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print()
    
    try:
        # Import and run enhanced analysis
        from enhanced_economic_forecasting import EnhancedEconomicForecasting
        
        # FRED API key from original code
        FRED_API_KEY = "7a74bbd246e54e7458184b0137db2311"
        
        # Initialize system
        print("Initializing enhanced forecasting system...")
        forecaster = EnhancedEconomicForecasting(FRED_API_KEY)
        
        # Step 1: Collect real data
        print("\n1. COLLECTING REAL DATA FROM OFFICIAL SOURCES")
        print("-" * 40)
        forecaster.collect_all_real_data()
        
        # Step 2: Build trade networks
        print("\n2. BUILDING TRADE NETWORKS FROM REAL DATA")
        print("-" * 40)
        forecaster.build_trade_networks_from_real_data()
        
        # Step 3: Calculate network features
        print("\n3. CALCULATING NETWORK TOPOLOGY FEATURES")
        print("-" * 40)
        forecaster.calculate_network_features()
        
        # Step 4: Prepare ML dataset
        print("\n4. PREPARING MACHINE LEARNING DATASET")
        print("-" * 40)
        dataset = forecaster.prepare_ml_dataset_with_real_data()
        print(f"Dataset shape: {dataset.shape}")
        print(f"Countries included: {sorted(dataset['country'].unique())}")
        print(f"Years covered: {dataset['year'].min()}-{dataset['year'].max()}")
        
        # Step 5: Run ML analysis
        print("\n5. RUNNING MACHINE LEARNING ANALYSIS")
        print("-" * 40)
        results = forecaster.run_enhanced_ml_analysis()
        
        # Print results summary
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        for model_name, metrics in results.items():
            print(f"{model_name:20s} | R²: {metrics['r2']:.3f} | RMSE: {metrics['rmse']:.3f}")
        
        # Step 6: Generate visualizations
        print("\n6. GENERATING VISUALIZATIONS")
        print("-" * 40)
        forecaster.generate_enhanced_visualizations()
        
        # Step 7: Generate report
        print("\n7. GENERATING COMPREHENSIVE REPORT")
        print("-" * 40)
        report = forecaster.generate_enhanced_report()
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"End time: {datetime.now()}")
        print("\nOutput files created:")
        print("- Report: ../reports/enhanced_real_data_report.md")
        print("- Dataset: ../data/real_ml_dataset.csv")
        print("- Visualizations: ../figures/real_data_charts/")
        print("- Logs: ../logs/real_data_run.log")
        
        # Key findings
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        print(f"\nBest performing model: {best_model} (R² = {results[best_model]['r2']:.3f})")
        
        # Data quality summary
        if hasattr(forecaster, 'ml_dataset'):
            missing_pct = forecaster.ml_dataset.isnull().sum().sum() / forecaster.ml_dataset.size * 100
            print(f"Data completeness: {100 - missing_pct:.1f}%")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: Analysis failed - {e}")
        print("Check ../logs/real_data_run.log for details")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())