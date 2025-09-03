# TheQA Research for Computational Problems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

# SR Earthquake Prediction System (boil5.py)

## Overview

This system implements macro-scale stochastic resonance analysis for earthquake risk assessment using real seismic data. The method identifies optimal noise levels that enhance signal detection in earthquake precursor patterns on yearly timescales.

## Key Features

- Real-time USGS earthquake data integration (up to 20 years)
- Gardner-Knopoff aftershock filtering
- Multi-parameter stress field calculation
- Stochastic resonance optimization across temporal scales
- Retrospective validation against historical events
- Comprehensive criticality analysis

## Scientific Method

### Data Processing
1. **Earthquake Catalog Loading**: Downloads real USGS data via FDSN web services
2. **Aftershock Filtering**: Implements Gardner-Knopoff windowing algorithm to separate mainshocks from aftershocks
3. **Stress Field Calculation**: Maps seismic energy release onto spatial grid using Gutenberg-Richter scaling
4. **Multi-source Integration**: Incorporates GPS station data where available

### Stochastic Resonance Analysis
- Tests noise levels from 10^-4 to 1.0 across five temporal scales (hourly to yearly)
- Calculates Signal-to-Noise Ratio (SNR) for each scale-noise combination
- Identifies optimal parameters maximizing system response

### Validation Methods
- **Retrospective Testing**: Evaluates performance against M6+ historical California earthquakes
- **Statistical Analysis**: Uses percentile ranking to assess prediction accuracy
- **Cross-validation**: Tests consistency across different time periods

## Results Summary (California 2015-2025)

### Data Statistics
- **Total Events**: 42,772 earthquakes (M≥2.0)
- **Mainshocks**: 9,494 (22.2% of catalog)
- **Aftershocks**: 33,278 (77.8% of catalog, filtered)
- **Magnitude Range**: 2.0 - 7.1

### Optimal Parameters
- **Scale**: Yearly (365-day aggregation)
- **Noise Level**: 0.049417 (42% of theoretical critical value)
- **SNR**: 5.639

### Criticality Metrics
- **Criticality Index**: 832.9 (CRITICAL level)
- **b-value**: 1.576 (Gutenberg-Richter slope)
- **Correlation Length**: 275 km
- **Hurst Exponent**: 0.787 (long-range correlations)

### Validation Results
- **Ridgecrest 2019 (M7.1)**: 92.5th percentile risk classification
- **Success Rate**: 100% for tested M6+ events (limited sample)

## Physical Interpretation

### Yearly-Scale Dominance
The system exhibits maximum stochastic resonance at yearly timescales, suggesting:
- Earthquake processes operate on geological timescales
- Monthly-to-yearly stress accumulation patterns are detectable
- Short-term prediction may be inherently limited by this temporal constraint

### Quiescence Patterns
- **30-day**: Z-score = -2.858 (recent activity increase)
- **90-day**: Z-score = +2.809 (moderate decrease)
- **365-day**: Z-score = +3.000 (significant long-term decrease)

This pattern suggests recent activation following extended quiescence.

## Installation & Usage

### Requirements

​
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
requests>=2.25.0
scikit-learn>=1.0.0
tqdm>=4.62.0

### Basic Usage
```python
python boil5.py
```
​
### The system will:
- Load 10 years of California earthquake data
- Apply Gardner-Knopoff filtering
- Calculate stress fields and run SR analysis
- Generate visualization and save results
Output Files
- california_production_analysis.png: Multi-panel scientific visualization
- california_production_results.json: Numerical results in JSON format

### Limitations
- Geographic Scope: Currently optimized for California seismicity
- GPS Integration: Limited to available station data
- Prediction Window: Provides risk assessment on monthly timescales, not precise event timing
- Sample Size: Validation limited by number of large historical earthquakes

### Technical Notes
Aftershock Filtering
- Uses Gardner-Knopoff space-time windows:
- Time window: T = 155 × 10^(0.013×M) days for M<6.5
- Space window: L = 10^(0.035×M + 0.965) km for M<6.5

b-value Calculation
- Maximum likelihood estimation with magnitude completeness testing:
- b = 1 / (mean_magnitude - Mc + 0.05)

Corrected for finite sample bias using Aki's method.

### Risk Assessment
Criticality Index combines:
- Stress field susceptibility (variance)
- Spatial correlation length
- Temporal correlations (Hurst exponent)
- Seismicity rate changes (b-value)
- Quiescence anomalies

### Possible Future Work
- Extension to other seismic regions
- Integration of additional geophysical data (InSAR, gravity)
- Real-time monitoring implementation
- Ensemble modeling with multiple algorithms
- Uncertainty quantification

### **License**
- This project follows a dual-license model:

- For Personal & Research Use: CC BY-NC 4.0 → Free for non-commercial use only.
- For Commercial Use: Companies must obtain a commercial license (Elastic License 2.0).

📜 For details, see the LICENSE file.


### ***Contributors***

- Matthias - Human resources
- Arti Cyan - Artificial  resources


### ***Contact & Support***

- For inquiries regarding commercial licensing or support, please contact:📧 theqa@posteo.com 🌐 www.theqa.space 🚀🚀🚀

- 🚀 Get started with TheQA and explore new frontiers in optimization! 🚀

---
![SIZEMO](https://github.com/hermannhart/sizemo/sizemo.jpg)