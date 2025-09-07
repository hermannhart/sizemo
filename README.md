# TheQA Research for Computational Problems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

# SIZEMO SR Earthquake Prediction System
![SIZEMO](https://github.com/hermannhart/sizemo/blob/main/sizemo2.jpg)

Evolution to Version 8: 
What started as a simple observation about boiling water has evolved into a possible earthquake prediction system achieving F1-scores of 0.90 in Japan through the application of stochastic resonance to seismic systems.

## The Core Innovation: Stochastic Resonance in Seismic Systems
### Possible Physics Behind SIZEMO (vague!)
The Earth's crust behaves like a self-organized critical system, accumulating stress until it reaches criticality. We discovered that this process contains a weak yearly-scale signal that can be amplified through stochastic resonance:
```bash
stress_field_sr = stress_field + np.random.normal(0, σ_c, shape)
```
The Formula behind it.

Where σ_c is the optimal noise level that maximizes signal detection:
- Japan: σ_c = 0.006 (subduction zone)
. California: σ_c = 0.0006 (transform fault)
- Chile: σ_c = 0.002 (subduction zone)

## Key Discoveries in Version 8

- Adaptive Decision Thresholds: Instead of a naive 0.5 threshold, we optimize for each region and magnitude
- Magnitude-Dependent Parameters: Different σ_c values for M4.5+, M5.0+, M5.5+
- Regional Time Windows: 365 days for subduction zones, 30 days for transform faults
- Composite Scoring: Balancing AUC (40%), F1-Score (40%), and Information Gain (20%)

## Performance Metrics (40 Years Data, 80/20 Split)
### Japan - Enough Data
M4.5+ Events (n=13,987 training, 934 test):
- F1-Score: 0.904 [95% CI: 0.887-0.922]
- Precision: 0.942
- Recall: 0.869
- AUC: 0.882

### Chile - Strong Validation
M4.5+ Events (n=15,108 training, 876 test):
- F1-Score: 0.876 [95% CI: 0.859-0.895]
- Precision: 0.849
- Recall: 0.906
- AUC: 0.878

### California - Honest Limitations
M4.5+ Events (n=702 training, 582 test):
- F1-Score: 0.582 [95% CI: 0.394-0.688]
- Precision: 0.495
- Recall: 0.710
- AUC: 0.857

## Technical Implementation
Core Algorithm (Simplified)
```python
def predict_earthquakes(catalog, σ_c, time_window, threshold):
    # 1. Decluster using Gardner-Knopoff
    mainshocks = gardner_knopoff_decluster(catalog)
    
    # 2. Calculate stress field
    stress_field = calculate_stress_accumulation(mainshocks, time_window)
    
    # 3. Apply stochastic resonance
    enhanced_field = stress_field + np.random.normal(0, σ_c, stress_field.shape)
    
    # 4. Threshold for predictions
    predictions = enhanced_field > threshold
    
    return predictions
```

## Key Components

- Gardner-Knopoff Declustering: Removes aftershocks/foreshocks to isolate mainshocks
- Hanks-Kanamori Energy Relation: E = 10^(1.44M + 5.24) for realistic energy distribution
 -Adaptive Gaussian Filtering: Spatial smoothing adapted to grid resolution
- Bootstrap Validation: 50 iterations for robust confidence intervals
- Temporal Cross-Validation: 5-fold time series split

## Installation and Usage
```bash
# Clone repository
git clone https://github.com/hermannhart/sizemo.git

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib requests

# Run analysis
python sizemo8.py
```

## Validation Methodology
### No Data Leakage Guaranteed

Training data strictly before cutoff date
90-day buffer between training and test periods
Stress calculations use only past data
No future information in any calculation

### Statistical Robustness

Bootstrap with 50 iterations
95% confidence intervals for all metrics
Cross-validation across time periods
Multiple magnitude thresholds tested

### Honest Limitations

- Regional Calibration Required: σ_c must be optimized for each tectonic setting
- Transform Faults Challenge: California shows weaker performance (F1~0.58)
- Sample Size for Large Events: M5.5+ events have limited samples for robust statistics
- No Prospective Testing Yet: All validation is retrospective

### Why This Could Works: The Stochastic Resonance Hypothesis
Traditional approaches treat seismic noise as a problem to be filtered. We discovered it's actually the key to prediction:

- The Earth's crust has a weak periodic signal (yearly scale)
- This signal is buried in noise and undetectable directly
- Adding optimal noise (σ_c) creates stochastic resonance
- The signal becomes detectable through noise-enhanced transitions

This explains why deterministic models have limited success - they fight the noise instead of using it.

## Future Directions

- Prospective Testing: Real-time predictions for next 6 months
- Global Application: Optimize σ_c for all seismic regions
- Physical Understanding: Why yearly scale? Seasonal loading?
- Integration: Combine with existing operational systems

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
![SIZEMO](https://github.com/hermannhart/sizemo/blob/main/sizemo.jpg)
