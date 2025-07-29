# Real Economic Data Sources Documentation

## Overview
This document details the actual economic data sources used in the enhanced replication, replacing simulated data with real indicators from official sources.

## Data Sources

### 1. **Federal Reserve Economic Data (FRED)**
- **API Key Required**: Yes (included in code)
- **Base URL**: https://api.stlouisfed.org/fred/
- **Update Frequency**: Daily to Monthly
- **Coverage**: 1947-present for most US series

**Key Indicators Retrieved:**
- GDPC1: Real Gross Domestic Product (US)
- A191RL1Q225SBEA: Real GDP Growth Rate (US)
- UNRATE: Unemployment Rate (US)
- CPIAUCSL: Consumer Price Index (US)
- DTWEXBGS: Trade Weighted US Dollar Index
- VIXCLS: CBOE Volatility Index
- DCOILWTICO: Crude Oil Prices (WTI)
- GEPUCURRENT: Global Economic Policy Uncertainty Index
- International GDP growth rates for major economies

### 2. **World Bank Open Data**
- **API Key Required**: No (public access)
- **Python Package**: wbgapi
- **Update Frequency**: Annual
- **Coverage**: 1960-present for most countries

**Key Indicators Retrieved:**
- NY.GDP.MKTP.KD.ZG: GDP growth (annual %)
- NY.GDP.MKTP.CD: GDP (current US$)
- SP.POP.TOTL: Population, total
- FP.CPI.TOTL.ZG: Inflation, consumer prices (annual %)
- NE.EXP.GNFS.ZS: Exports of goods and services (% of GDP)
- NE.IMP.GNFS.ZS: Imports of goods and services (% of GDP)
- BX.KLT.DINV.WD.GD.ZS: Foreign direct investment, net inflows (% of GDP)

### 3. **UN Comtrade Database**
- **API Key Required**: No (rate limited)
- **Base URL**: https://comtradeapi.un.org/
- **Limitations**: 100 requests/hour (free tier)
- **Coverage**: 1962-present

**Data Retrieved:**
- Bilateral trade flows (exports/imports)
- Product-level trade data (HS codes)
- Annual trade values in USD

**Note**: Due to API limitations, only sample years are fetched. For production:
- Use UN Comtrade subscription for bulk downloads
- Consider WITS (World Integrated Trade Solution) as alternative
- Implement caching mechanism for historical data

### 4. **OECD Statistics**
- **Access Method**: pandas_datareader
- **Coverage**: OECD member countries
- **Frequency**: Quarterly/Annual

**Key Indicators:**
- GDP growth rates
- Consumer Price Index
- Unemployment rates
- Long-term interest rates

### 5. **Yahoo Finance**
- **API Key Required**: No
- **Python Package**: yfinance
- **Update Frequency**: Daily
- **Coverage**: Varies by index

**Stock Market Indices Retrieved:**
- ^GSPC: S&P 500 (USA)
- ^IXIC: NASDAQ Composite (USA)
- ^FTSE: FTSE 100 (UK)
- ^GDAXI: DAX (Germany)
- ^N225: Nikkei 225 (Japan)
- ^HSI: Hang Seng (Hong Kong)
- 000001.SS: Shanghai Composite (China)
- ^BSESN: BSE SENSEX (India)
- ^KS11: KOSPI (South Korea)
- ^BVSP: Bovespa (Brazil)

## Data Quality Considerations

### Missing Data Handling
1. **Forward Fill**: For gaps up to 2 years
2. **Backward Fill**: For initial year gaps
3. **Interpolation**: For quarterly to annual conversion
4. **Exclusion**: Countries with >30% missing data

### Data Lags
- World Bank: 6-12 month publication lag
- UN Comtrade: 3-6 month reporting lag
- FRED: 1-3 month for most series
- Stock indices: Real-time

### Validation Steps
1. Cross-reference values across sources
2. Check for outliers and anomalies
3. Verify temporal consistency
4. Validate against published reports

## Implementation Notes

### API Rate Limits
- UN Comtrade: 100 requests/hour
- World Bank: 50 requests/second
- FRED: 120 requests/minute
- Yahoo Finance: No official limit

### Recommended Enhancements
1. **Caching Strategy**: Store historical data locally
2. **Batch Processing**: Group API requests efficiently
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Data Updates**: Schedule regular updates for recent periods

### Alternative Data Sources
For more comprehensive coverage:
- **IMF Data**: International Financial Statistics
- **BIS Statistics**: Bank settlements and FX data
- **ECB Data Warehouse**: European economic indicators
- **National Statistical Offices**: Country-specific data

## Usage Example

```python
from real_data_collection import RealDataCollector

# Initialize collector
collector = RealDataCollector()

# Fetch World Bank data
wb_data = collector.fetch_world_bank_data(
    start_year=2010,
    end_year=2022
)

# Build trade networks
trade_matrix = collector.build_trade_matrix(
    year=2020,
    countries=['USA', 'CHN', 'DEU', 'JPN']
)
```

## Data Dictionary

Full data dictionary available in: `../data/real_data/data_dictionary.csv`

## Contact

For data access issues or questions:
- World Bank: data@worldbank.org
- UN Comtrade: comtrade@un.org
- FRED: fred.help@stls.frb.org