"""
SIZEMO EARTHQUAKE PREDICTION SYSTEM - ADVANCED OPTIMIZATION VERSION
====================================================================
Version 8.0 - Vollständige Parameter-Optimierung mit korrekter Threshold-Anpassung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import signal, stats, optimize
from scipy.special import gammaln
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
from tqdm import tqdm
import json
import logging
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
import multiprocessing

warnings.filterwarnings('ignore')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedOptimizedPredictor:
    """
    Vollständig optimiertes Erdbebenvorhersagesystem mit automatischer Parameter-Anpassung
    """
    
    def __init__(self, region_bounds, grid_resolution_km=10):
        self.region_bounds = region_bounds
        self.grid_resolution = grid_resolution_km
        
        # Räumliches Grid
        self.n_lat = int((region_bounds['lat_max'] - region_bounds['lat_min']) * 111 / grid_resolution_km)
        self.n_lon = int((region_bounds['lon_max'] - region_bounds['lon_min']) * 111 / grid_resolution_km)
        
        self.lat_grid = np.linspace(region_bounds['lat_min'], region_bounds['lat_max'], self.n_lat)
        self.lon_grid = np.linspace(region_bounds['lon_min'], region_bounds['lon_max'], self.n_lon)
        
        logger.info(f"System initialisiert: {self.n_lat}x{self.n_lon} Grid ({grid_resolution_km}km Auflösung)")
        
        # Optimierte Parameter (werden automatisch gefunden)
        self.optimal_params = {}
        
    def load_extended_catalog(self, start_date, end_date, min_magnitude=2.0):
        """Lade erweiterten Katalog (40+ Jahre für Robustheit)"""
        logger.info(f"Lade Katalog: {start_date} bis {end_date}, M≥{min_magnitude}")
        
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        all_events = []
        chunk_size = timedelta(days=365)
        
        current_start = start_dt
        years_loaded = 0
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            
            params = {
                'format': 'csv',
                'starttime': current_start.strftime('%Y-%m-%d'),
                'endtime': current_end.strftime('%Y-%m-%d'),
                'minmagnitude': min_magnitude,
                'minlatitude': self.region_bounds['lat_min'],
                'maxlatitude': self.region_bounds['lat_max'],
                'minlongitude': self.region_bounds['lon_min'],
                'maxlongitude': self.region_bounds['lon_max'],
                'orderby': 'time'
            }
            
            try:
                response = requests.get(
                    "https://earthquake.usgs.gov/fdsnws/event/1/query",
                    params=params,
                    timeout=120
                )
                response.raise_for_status()
                
                chunk_data = pd.read_csv(StringIO(response.text))
                chunk_data['time'] = pd.to_datetime(chunk_data['time'])
                
                if chunk_data['time'].dt.tz is None:
                    chunk_data['time'] = chunk_data['time'].dt.tz_localize('UTC')
                else:
                    chunk_data['time'] = chunk_data['time'].dt.tz_convert('UTC')
                
                all_events.append(chunk_data)
                years_loaded += 1
                logger.info(f"  Jahr {years_loaded} geladen: {len(chunk_data)} Events")
                
            except Exception as e:
                logger.warning(f"Fehler beim Laden: {e}")
            
            current_start = current_end
        
        if all_events:
            catalog = pd.concat(all_events, ignore_index=True)
            catalog = catalog.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])
            catalog = catalog.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Gesamt: {len(catalog)} Events (M{catalog['mag'].min():.1f}-{catalog['mag'].max():.1f})")
            
            self.catalog = catalog
            self.original_catalog = catalog.copy()
            
            return catalog
        
        return None
    
    def optimize_regional_parameters_advanced(self, catalog, magnitude_thresholds=[4.5, 5.0, 5.5]):
        """
        Erweiterte Optimierung mit Fokus auf Precision UND Recall
        """
        logger.info("Starte erweiterte Parameter-Optimierung...")
        
        optimal_params = {}
        
        # Erweiterte Parameter-Suchräume
        sigma_c_range = np.logspace(-4, -0.5, 20)  # Feinere Auflösung: 0.0001 bis 0.316
        time_window_range = [30, 60, 90, 120, 180, 270, 365]  # Mehr Optionen
        threshold_range = np.linspace(0.1, 0.9, 9)  # Optimiere auch den Decision Threshold
        
        for mag_threshold in magnitude_thresholds:
            logger.info(f"  Optimiere für M≥{mag_threshold}")
            
            best_score = -np.inf
            best_params = None
            
            # Grid Search über alle Parameter
            for sigma_c in sigma_c_range:
                for time_window in time_window_range:
                    for decision_threshold in threshold_range:
                        # Evaluiere diese Parameter-Kombination
                        score = self._evaluate_parameters_advanced(
                            catalog, sigma_c, time_window, decision_threshold, mag_threshold
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'sigma_c': sigma_c,
                                'time_window': time_window,
                                'decision_threshold': decision_threshold,
                                'score': score
                            }
            
            # Fein-Tuning um beste Parameter herum
            if best_params:
                fine_tuned = self._fine_tune_parameters(
                    catalog, best_params, mag_threshold
                )
                optimal_params[f'M{mag_threshold}+'] = fine_tuned
                
                logger.info(f"    Beste Parameter: σ_c={fine_tuned['sigma_c']:.5f}, "
                           f"time={fine_tuned['time_window']}d, "
                           f"threshold={fine_tuned['decision_threshold']:.3f}, "
                           f"score={fine_tuned['score']:.3f}")
            else:
                # Fallback auf vernünftige Defaults
                optimal_params[f'M{mag_threshold}+'] = {
                    'sigma_c': 0.01,
                    'time_window': 90,
                    'decision_threshold': 0.5,
                    'score': 0
                }
        
        self.optimal_params = optimal_params
        return optimal_params
    
    def _evaluate_parameters_advanced(self, catalog, sigma_c, time_window, threshold, mag_threshold):
        """
        Erweiterte Evaluation mit F1-Score statt nur Info Gain
        """
        # Teile Daten in Train/Test (80/20)
        split_time = catalog['time'].min() + \
                    pd.Timedelta(days=int((catalog['time'].max() - catalog['time'].min()).days * 0.8))
        
        train = catalog[catalog['time'] < split_time]
        test = catalog[catalog['time'] >= split_time]
        
        test_events = test[test['mag'] >= mag_threshold]
        
        if len(test_events) < 10:
            return -np.inf
        
        # Berechne Stress-Feld mit diesen Parametern
        stress_field = self._calculate_stress_field_optimized(train, time_window)
        
        # Füge SR-Noise hinzu (mehrere Realisierungen für Stabilität)
        predictions_all = []
        actuals_all = []
        
        for _ in range(5):  # 5 Realisierungen für robustere Schätzung
            stress_field_sr = stress_field + np.random.normal(0, sigma_c, stress_field.shape)
            stress_field_sr = np.clip(stress_field_sr, 0, 1)  # Normalisiere auf [0,1]
            
            predictions = []
            actuals = []
            
            # Positive samples
            for _, event in test_events.iterrows():
                lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
                lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
                
                if 0 <= lat_idx < self.n_lat and 0 <= lon_idx < self.n_lon:
                    predictions.append(stress_field_sr[lat_idx, lon_idx])
                    actuals.append(1)
            
            # Negative samples (adaptiv)
            n_negatives = min(len(test_events) * 5, 500)  # Cap bei 500
            
            for _ in range(n_negatives):
                lat_idx = np.random.randint(0, self.n_lat)
                lon_idx = np.random.randint(0, self.n_lon)
                predictions.append(stress_field_sr[lat_idx, lon_idx])
                actuals.append(0)
            
            predictions_all.extend(predictions)
            actuals_all.extend(actuals)
        
        if len(np.unique(actuals_all)) > 1:
            # Berechne Metriken
            auc = roc_auc_score(actuals_all, predictions_all)
            
            # Binäre Vorhersagen mit optimiertem Threshold
            binary_pred = [1 if p > threshold else 0 for p in predictions_all]
            
            # Confusion Matrix
            try:
                tn, fp, fn, tp = confusion_matrix(actuals_all, binary_pred).ravel()
            except:
                return -np.inf
            
            # Berechne Precision, Recall und F1-Score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Kombinierter Score: Gewichtung von AUC, F1 und Info Gain
            info_gain = (auc - 0.5) * 2
            
            # Composite Score (anpassbar je nach Priorität)
            score = 0.4 * auc + 0.4 * f1 + 0.2 * info_gain
            
            # Bonus für hohe Precision (wichtig für Erdbebenvorhersage)
            if precision > 0.7:
                score += 0.1
            
            return score
        
        return -np.inf
    
    def _fine_tune_parameters(self, catalog, initial_params, mag_threshold):
        """
        Fein-Tuning der Parameter um das initiale Optimum herum
        """
        best_params = initial_params.copy()
        
        # Feinere Suche um beste Parameter
        sigma_c_fine = np.linspace(
            initial_params['sigma_c'] * 0.5, 
            initial_params['sigma_c'] * 2, 
            10
        )
        
        threshold_fine = np.linspace(
            max(0.1, initial_params['decision_threshold'] - 0.1),
            min(0.9, initial_params['decision_threshold'] + 0.1),
            10
        )
        
        best_score = initial_params['score']
        
        for sigma_c in sigma_c_fine:
            for threshold in threshold_fine:
                score = self._evaluate_parameters_advanced(
                    catalog, 
                    sigma_c, 
                    initial_params['time_window'],
                    threshold,
                    mag_threshold
                )
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'sigma_c': sigma_c,
                        'time_window': initial_params['time_window'],
                        'decision_threshold': threshold,
                        'score': score
                    }
        
        return best_params
    
    def _calculate_stress_field_optimized(self, catalog, time_window):
        """Optimierte Stress-Feld Berechnung"""
        stress_field = np.zeros((self.n_lat, self.n_lon))
        
        end_time = catalog['time'].max()
        start_time = end_time - pd.Timedelta(days=time_window)
        recent_events = catalog[(catalog['time'] >= start_time)]
        
        # Verwende magnitude-abhängige Gewichtung
        for _, event in recent_events.iterrows():
            lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
            lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
            
            if 0 <= lat_idx < self.n_lat and 0 <= lon_idx < self.n_lon:
                # Optimierte Energie-Berechnung
                energy = 10 ** (1.44 * event['mag'] + 5.24)  # Hanks-Kanamori Relation
                
                # Räumliche Verteilung mit magnitude-abhängigem Radius
                radius = max(1, int(event['mag'] - 3))  # Größere Events haben größeren Einfluss
                sigma = radius * 2
                
                for i in range(max(0, lat_idx-radius*3), min(self.n_lat, lat_idx+radius*3+1)):
                    for j in range(max(0, lon_idx-radius*3), min(self.n_lon, lon_idx+radius*3+1)):
                        dist = np.sqrt((i-lat_idx)**2 + (j-lon_idx)**2)
                        weight = np.exp(-dist**2 / (2*sigma**2))
                        stress_field[i, j] += energy * weight
        
        # Normalisierung mit Logarithmus
        if stress_field.max() > 0:
            stress_field = np.log10(stress_field + 1)
            stress_field /= stress_field.max()
        
        # Adaptive Gaussian Filterung
        sigma_filter = max(1, min(3, self.grid_resolution / 5))
        return gaussian_filter(stress_field, sigma=sigma_filter)
    
    def perform_robust_validation(self, n_bootstrap=50, test_years=5):
        """
        Robuste Validierung mit optimierten Parametern
        """
        logger.info(f"Starte robuste Validierung mit {n_bootstrap} Bootstrap-Iterationen...")
        
        bootstrap_results = {
            'M4.5+': {'aucs': [], 'precisions': [], 'recalls': [], 'f1s': []},
            'M5.0+': {'aucs': [], 'precisions': [], 'recalls': [], 'f1s': []},
            'M5.5+': {'aucs': [], 'precisions': [], 'recalls': [], 'f1s': []}
        }
        
        for i in range(n_bootstrap):
            if i % 10 == 0:
                logger.info(f"  Bootstrap Iteration {i+1}/{n_bootstrap}")
            
            # Resample mit Replacement
            resampled_catalog = self.catalog.sample(n=len(self.catalog), replace=True)
            resampled_catalog = resampled_catalog.sort_values('time').reset_index(drop=True)
            
            # Validiere für jede Magnitude
            for mag_key in bootstrap_results.keys():
                mag_threshold = float(mag_key.replace('M', '').replace('+', ''))
                
                # Verwende optimierte Parameter
                if mag_key in self.optimal_params:
                    params = self.optimal_params[mag_key]
                    
                    # Validiere mit korrektem Threshold
                    result = self._validate_with_optimal_threshold(
                        resampled_catalog, 
                        mag_threshold,
                        params['sigma_c'],
                        params['time_window'],
                        params['decision_threshold']
                    )
                    
                    if result:
                        bootstrap_results[mag_key]['aucs'].append(result['auc'])
                        bootstrap_results[mag_key]['precisions'].append(result['precision'])
                        bootstrap_results[mag_key]['recalls'].append(result['recall'])
                        bootstrap_results[mag_key]['f1s'].append(result['f1'])
        
        # Berechne Konfidenzintervalle
        confidence_intervals = {}
        
        for mag_key, metrics in bootstrap_results.items():
            ci = {}
            for metric_name, values in metrics.items():
                if values:
                    ci[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'ci_95': (np.percentile(values, 2.5), np.percentile(values, 97.5))
                    }
            confidence_intervals[mag_key] = ci
        
        self.bootstrap_results = confidence_intervals
        return confidence_intervals
    
    def _validate_with_optimal_threshold(self, catalog, mag_threshold, sigma_c, time_window, threshold):
        """Validierung mit optimiertem Threshold"""
        # Split 80/20
        split_time = catalog['time'].min() + \
                    pd.Timedelta(days=int((catalog['time'].max() - catalog['time'].min()).days * 0.8))
        
        train = catalog[catalog['time'] < split_time]
        test = catalog[catalog['time'] >= split_time]
        
        test_events = test[test['mag'] >= mag_threshold]
        
        if len(test_events) < 5:
            return None
        
        # Stress-Feld
        stress_field = self._calculate_stress_field_optimized(train, time_window)
        stress_field_sr = stress_field + np.random.normal(0, sigma_c, stress_field.shape)
        stress_field_sr = np.clip(stress_field_sr, 0, 1)
        
        # Vorhersagen
        predictions = []
        actuals = []
        
        for _, event in test_events.iterrows():
            lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
            lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
            
            if 0 <= lat_idx < self.n_lat and 0 <= lon_idx < self.n_lon:
                predictions.append(stress_field_sr[lat_idx, lon_idx])
                actuals.append(1)
        
        # Negative samples (adaptiv)
        n_negatives = min(len(test_events) * 5, 500)
        
        negative_samples = []
        attempts = 0
        max_attempts = n_negatives * 10
        
        while len(negative_samples) < n_negatives and attempts < max_attempts:
            lat_idx = np.random.randint(0, self.n_lat)
            lon_idx = np.random.randint(0, self.n_lon)
            
            # Stelle sicher, dass negative samples nicht zu nah an echten Events sind
            lat = self.lat_grid[lat_idx]
            lon = self.lon_grid[lon_idx]
            
            min_dist = float('inf')
            for _, event in test_events.iterrows():
                dist = self._haversine_distance(lat, lon, event['latitude'], event['longitude'])
                min_dist = min(min_dist, dist)
            
            if min_dist > 20:  # 20km Mindestabstand
                negative_samples.append((lat_idx, lon_idx))
                predictions.append(stress_field_sr[lat_idx, lon_idx])
                actuals.append(0)
            
            attempts += 1
        
        if len(np.unique(actuals)) > 1 and len(predictions) > 10:
            auc = roc_auc_score(actuals, predictions)
            
            # Binäre Vorhersagen mit optimiertem Threshold
            binary_pred = [1 if p > threshold else 0 for p in predictions]
            
            try:
                tn, fp, fn, tp = confusion_matrix(actuals, binary_pred).ravel()
            except:
                return None
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return None
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Haversine Distanz in km"""
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def perform_temporal_cross_validation(self, n_splits=5):
        """
        Zeitliche Cross-Validation mit optimierten Parametern
        """
        logger.info(f"Starte zeitliche Cross-Validation mit {n_splits} Splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(self.catalog)):
            logger.info(f"  Cross-Validation Split {i+1}/{n_splits}")
            
            train = self.catalog.iloc[train_idx]
            test = self.catalog.iloc[test_idx]
            
            split_results = {}
            
            for mag_threshold in [4.5, 5.0, 5.5]:
                if f'M{mag_threshold}+' in self.optimal_params:
                    params = self.optimal_params[f'M{mag_threshold}+']
                    
                    result = self._validate_with_optimal_threshold(
                        pd.concat([train, test]),
                        mag_threshold,
                        params['sigma_c'],
                        params['time_window'],
                        params['decision_threshold']
                    )
                    
                    if result:
                        split_results[f'M{mag_threshold}+'] = result
            
            cv_results.append(split_results)
        
        # Aggregiere Ergebnisse
        aggregated = {}
        
        for mag_key in ['M4.5+', 'M5.0+', 'M5.5+']:
            aucs = [r[mag_key]['auc'] for r in cv_results if mag_key in r]
            precisions = [r[mag_key]['precision'] for r in cv_results if mag_key in r]
            recalls = [r[mag_key]['recall'] for r in cv_results if mag_key in r]
            f1s = [r[mag_key]['f1'] for r in cv_results if mag_key in r]
            
            if aucs:
                aggregated[mag_key] = {
                    'auc_mean': np.mean(aucs),
                    'auc_std': np.std(aucs),
                    'precision_mean': np.mean(precisions),
                    'precision_std': np.std(precisions),
                    'recall_mean': np.mean(recalls),
                    'recall_std': np.std(recalls),
                    'f1_mean': np.mean(f1s),
                    'f1_std': np.std(f1s)
                }
        
        self.cv_results = aggregated
        return aggregated
    
    def create_comprehensive_report(self):
        """Erstelle umfassenden wissenschaftlichen Report"""
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Optimierte Parameter
        ax1 = fig.add_subplot(gs[0, 0:2])
        param_data = []
        for mag_key, params in self.optimal_params.items():
            param_data.append({
                'Magnitude': mag_key,
                'σ_c': f"{params['sigma_c']:.5f}",
                'Time (days)': params['time_window'],
                'Threshold': f"{params['decision_threshold']:.2f}",
                'Score': f"{params['score']:.3f}"
            })
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            ax1.axis('off')
            table = ax1.table(cellText=param_df.values,
                             colLabels=param_df.columns,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.15, 0.2, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax1.set_title('Optimierte Parameter (mit Decision Threshold)', fontsize=12, fontweight='bold')
        
        # 2. Bootstrap AUC mit Konfidenzintervallen
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        if hasattr(self, 'bootstrap_results'):
            mag_keys = list(self.bootstrap_results.keys())
            x = np.arange(len(mag_keys))
            
            # AUC mit Konfidenzintervallen
            means = [self.bootstrap_results[k]['aucs']['mean'] if 'aucs' in self.bootstrap_results[k] else 0 
                    for k in mag_keys]
            ci_lower = [self.bootstrap_results[k]['aucs']['ci_95'][0] if 'aucs' in self.bootstrap_results[k] else 0 
                       for k in mag_keys]
            ci_upper = [self.bootstrap_results[k]['aucs']['ci_95'][1] if 'aucs' in self.bootstrap_results[k] else 0 
                       for k in mag_keys]
            
            ax2.bar(x, means, yerr=[np.array(means)-np.array(ci_lower), 
                                    np.array(ci_upper)-np.array(means)],
                   capsize=10, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_xticks(x)
            ax2.set_xticklabels(mag_keys)
            ax2.set_ylabel('AUC')
            ax2.set_title('Bootstrap AUC mit 95% Konfidenzintervallen', fontsize=12)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
            ax2.set_ylim([0.4, 1.0])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Precision vs Recall Trade-off
        ax3 = fig.add_subplot(gs[1, 0:2])
        
        if hasattr(self, 'bootstrap_results'):
            colors = ['blue', 'orange', 'green']
            for i, mag_key in enumerate(self.bootstrap_results.keys()):
                if 'precisions' in self.bootstrap_results[mag_key] and 'recalls' in self.bootstrap_results[mag_key]:
                    precision = self.bootstrap_results[mag_key]['precisions']['mean']
                    recall = self.bootstrap_results[mag_key]['recalls']['mean']
                    
                    # Error bars
                    p_err = self.bootstrap_results[mag_key]['precisions']['std']
                    r_err = self.bootstrap_results[mag_key]['recalls']['std']
                    
                    ax3.errorbar(recall, precision, xerr=r_err, yerr=p_err, 
                               fmt='o', markersize=10, label=mag_key, color=colors[i],
                               capsize=5, capthick=2)
            
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title('Precision vs Recall (mit optimiertem Threshold)', fontsize=12)
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Iso-F1 Kurven
            f1_scores = np.arange(0.2, 1.0, 0.2)
            for f1_score in f1_scores:
                x = np.linspace(0.01, 1, 100)
                y = f1_score * x / (2 * x - f1_score)
                ax3.plot(x, y, '--', color='gray', alpha=0.3, linewidth=0.5)
                ax3.text(0.9, y[90], f'F1={f1_score:.1f}', fontsize=7, color='gray')
        
        # 4. F1-Score Comparison
        ax4 = fig.add_subplot(gs[1, 2:4])
        
        if hasattr(self, 'bootstrap_results'):
            mag_keys = list(self.bootstrap_results.keys())
            f1_means = [self.bootstrap_results[k]['f1s']['mean'] if 'f1s' in self.bootstrap_results[k] else 0 
                       for k in mag_keys]
            f1_stds = [self.bootstrap_results[k]['f1s']['std'] if 'f1s' in self.bootstrap_results[k] else 0 
                      for k in mag_keys]
            
            bars = ax4.bar(mag_keys, f1_means, yerr=f1_stds, capsize=10, 
                          color=['green' if f1 > 0.5 else 'orange' if f1 > 0.3 else 'red' 
                                for f1 in f1_means], alpha=0.7)
            ax4.set_ylabel('F1-Score')
            ax4.set_title('F1-Score Performance', fontsize=12)
            ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good')
            ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate')
            ax4.set_ylim([0, 1])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Füge Werte über den Balken hinzu
            for bar, mean, std in zip(bars, f1_means, f1_stds):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Cross-Validation Results
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        if hasattr(self, 'cv_results'):
            cv_data = []
            for mag_key, results in self.cv_results.items():
                cv_data.append({
                    'Magnitude': mag_key,
                    'AUC': f"{results['auc_mean']:.3f} ± {results['auc_std']:.3f}",
                    'Precision': f"{results['precision_mean']:.3f} ± {results['precision_std']:.3f}",
                    'Recall': f"{results['recall_mean']:.3f} ± {results['recall_std']:.3f}",
                    'F1': f"{results['f1_mean']:.3f} ± {results['f1_std']:.3f}"
                })
            
            if cv_data:
                cv_df = pd.DataFrame(cv_data)
                ax5.axis('off')
                table = ax5.table(cellText=cv_df.values,
                                 colLabels=cv_df.columns,
                                 cellLoc='center',
                                 loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax5.set_title('Cross-Validation Ergebnisse (Mean ± Std)', fontsize=12, fontweight='bold')
        
        # 6. Information Gain
        ax6 = fig.add_subplot(gs[2, 2:4])
        
        if hasattr(self, 'bootstrap_results'):
            mag_keys = list(self.bootstrap_results.keys())
            
            # Berechne Info Gain aus AUC
            info_gains = []
            for k in mag_keys:
                if 'aucs' in self.bootstrap_results[k]:
                    auc = self.bootstrap_results[k]['aucs']['mean']
                    info_gain = (auc - 0.5) * 2  # Einfache Info Gain Approximation
                    info_gains.append(info_gain)
                else:
                    info_gains.append(0)
            
            colors = ['green' if ig > 0.3 else 'yellow' if ig > 0 else 'red' for ig in info_gains]
            bars = ax6.bar(mag_keys, info_gains, color=colors, alpha=0.7, edgecolor='black')
            ax6.set_ylabel('Information Gain')
            ax6.set_title('Information Gain (basierend auf AUC)', fontsize=12)
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Signifikant')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            # Füge Werte über den Balken hinzu
            for bar, ig in zip(bars, info_gains):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.02,
                        f'{ig:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 7. Sample Size Analysis
        ax7 = fig.add_subplot(gs[3, :2])
        
        mag_thresholds = [4.0, 4.5, 5.0, 5.5, 6.0]
        event_counts = [len(self.catalog[self.catalog['mag'] >= m]) for m in mag_thresholds]
        
        bars = ax7.bar([f'M{m}+' for m in mag_thresholds], event_counts, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        ax7.set_ylabel('Anzahl Events')
        ax7.set_title('Verfügbare Events nach Magnitude', fontsize=12)
        ax7.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Minimum für robuste Statistik')
        ax7.legend()
        
        # Füge Zahlen über den Balken hinzu
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 8. Finale wissenschaftliche Bewertung
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        assessment = "FINALE WISSENSCHAFTLICHE BEWERTUNG\n" + "="*40 + "\n\n"
        
        # Überprüfe Performance
        if hasattr(self, 'bootstrap_results'):
            # Durchschnittliche Metriken
            avg_auc = np.mean([self.bootstrap_results[k]['aucs']['mean'] 
                             for k in self.bootstrap_results.keys() 
                             if 'aucs' in self.bootstrap_results[k]])
            avg_precision = np.mean([self.bootstrap_results[k]['precisions']['mean'] 
                                    for k in self.bootstrap_results.keys() 
                                    if 'precisions' in self.bootstrap_results[k]])
            avg_recall = np.mean([self.bootstrap_results[k]['recalls']['mean'] 
                                 for k in self.bootstrap_results.keys() 
                                 if 'recalls' in self.bootstrap_results[k]])
            avg_f1 = np.mean([self.bootstrap_results[k]['f1s']['mean'] 
                            for k in self.bootstrap_results.keys() 
                            if 'f1s' in self.bootstrap_results[k]])
            
            assessment += f"Durchschnittliche Performance:\n"
            assessment += f"  AUC: {avg_auc:.3f}\n"
            assessment += f"  Precision: {avg_precision:.3f}\n"
            assessment += f"  Recall: {avg_recall:.3f}\n"
            assessment += f"  F1-Score: {avg_f1:.3f}\n\n"
            
            # Bewertung
            if avg_f1 > 0.5 and avg_auc > 0.7:
                assessment += "✓ System zeigt starke Vorhersagefähigkeit\n"
                assessment += "✓ Optimierung erfolgreich\n"
                assessment += "✓ Bereit für wissenschaftliche Publikation\n"
            elif avg_f1 > 0.3 and avg_auc > 0.6:
                assessment += "✓ System zeigt moderate Vorhersagefähigkeit\n"
                assessment += "⚠ Weitere Optimierung empfohlen\n"
                assessment += "→ Erwäge zusätzliche Features\n"
            else:
                assessment += "⚠ Begrenzte Vorhersagefähigkeit\n"
                assessment += "→ Überprüfe Datenqualität\n"
                assessment += "→ Erweitere Feature-Set\n"
                assessment += "→ Teste alternative Modelle\n"
            
            assessment += "\nNächste Schritte:\n"
            if avg_f1 > 0.5:
                assessment += "  1. Prospektive Validierung starten\n"
                assessment += "  2. Paper-Draft vorbereiten\n"
                assessment += "  3. Kontakt mit Experten aufnehmen\n"
            else:
                assessment += "  1. Feature Engineering erweitern\n"
                assessment += "  2. Hyperparameter-Tuning verfeinern\n"
                assessment += "  3. Alternative Modelle testen\n"
        
        ax8.text(0.05, 0.95, assessment, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        region_name = "Optimized"
        if hasattr(self, 'region_name'):
            region_name = self.region_name
        
        plt.suptitle(f'SIZEMO - Advanced Optimization Analysis - {region_name}', 
                    fontsize=16, fontweight='bold')
        
        return fig


def run_advanced_optimized_analysis(region='california', years_back=40):
    """
    Führe erweiterte optimierte Analyse durch
    """
    
    regions = {
        'california': {
            'lat_min': 32.0, 'lat_max': 42.0,
            'lon_min': -125.0, 'lon_max': -114.0
        },
        'japan': {
            'lat_min': 30.0, 'lat_max': 46.0,
            'lon_min': 129.0, 'lon_max': 146.0
        },
        'chile': {
            'lat_min': -35.0, 'lat_max': -17.0,
            'lon_min': -74.0, 'lon_max': -69.0
        }
    }
    
    print("="*70)
    print("SIZEMO - ADVANCED OPTIMIZATION ANALYSIS")
    print("Mit vollständiger Parameter- und Threshold-Optimierung")
    print("="*70)
    print(f"Region: {region.upper()}")
    print(f"Datenperiode: {years_back} Jahre")
    print("="*70)
    
    # Initialisiere System
    system = AdvancedOptimizedPredictor(regions[region], grid_resolution_km=10)
    system.region_name = region.upper()
    
    # Lade erweiterte Daten
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*years_back)).strftime('%Y-%m-%d')
    
    catalog = system.load_extended_catalog(start_date, end_date, min_magnitude=2.0)
    
    if catalog is None:
        print("Fehler: Keine Daten geladen")
        return None
    
    # 1. Erweiterte Parameter-Optimierung
    print("\n1. ERWEITERTE PARAMETER-OPTIMIERUNG")
    optimal_params = system.optimize_regional_parameters_advanced(
        catalog, 
        magnitude_thresholds=[4.5, 5.0, 5.5]
    )
    
    # 2. Bootstrap-Validierung mit optimierten Thresholds
    print("\n2. BOOTSTRAP-VALIDIERUNG")
    bootstrap_results = system.perform_robust_validation(
        n_bootstrap=50,  # Reduziert für schnellere Ausführung
        test_years=5
    )
    
    # 3. Cross-Validation
    print("\n3. ZEITLICHE CROSS-VALIDATION")
    cv_results = system.perform_temporal_cross_validation(n_splits=5)
    
    # 4. Erstelle Report
    print("\n4. ERSTELLE WISSENSCHAFTLICHEN REPORT")
    fig = system.create_comprehensive_report()
    
    # Speichere Ergebnisse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sizemo_advanced_{region}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nReport gespeichert: {filename}")
    
    # Speichere optimierte Parameter
    params_file = f"optimal_params_advanced_{region}_{timestamp}.json"
    with open(params_file, 'w') as f:
        # Konvertiere für JSON-Serialisierung
        json_params = {}
        for k, v in optimal_params.items():
            json_params[k] = {
                'sigma_c': float(v['sigma_c']),
                'time_window': int(v['time_window']),
                'decision_threshold': float(v['decision_threshold']),
                'score': float(v['score'])
            }
        
        json.dump({
            'region': region,
            'parameters': json_params,
            'bootstrap_results': {
                k: {
                    metric: {
                        'mean': float(v[metric]['mean']) if metric in v else None,
                        'std': float(v[metric]['std']) if metric in v and 'std' in v[metric] else None,
                        'ci_95': [float(v[metric]['ci_95'][0]), float(v[metric]['ci_95'][1])] 
                                if metric in v and 'ci_95' in v[metric] else None
                    } for metric in ['aucs', 'precisions', 'recalls', 'f1s']
                } for k, v in bootstrap_results.items()
            } if bootstrap_results else {},
            'cv_results': cv_results
        }, f, indent=2)
    
    print(f"Parameter gespeichert: {params_file}")
    
    # Zusammenfassung
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    
    for mag_key, params in optimal_params.items():
        print(f"\n{mag_key}:")
        print(f"  Optimale σ_c: {params['sigma_c']:.5f}")
        print(f"  Optimales Zeitfenster: {params['time_window']} Tage")
        print(f"  Optimaler Threshold: {params['decision_threshold']:.3f}")
        print(f"  Composite Score: {params['score']:.3f}")
    
    if bootstrap_results:
        print("\n" + "="*70)
        print("PERFORMANCE METRIKEN (95% CI):")
        print("="*70)
        for mag_key, results in bootstrap_results.items():
            print(f"\n{mag_key}:")
            if 'aucs' in results:
                ci = results['aucs']['ci_95']
                print(f"  AUC: {results['aucs']['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            if 'precisions' in results:
                ci = results['precisions']['ci_95']
                print(f"  Precision: {results['precisions']['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            if 'recalls' in results:
                ci = results['recalls']['ci_95']
                print(f"  Recall: {results['recalls']['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            if 'f1s' in results:
                ci = results['f1s']['ci_95']
                print(f"  F1-Score: {results['f1s']['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    plt.show()
    
    return system


def run_all_regions_advanced():
    """Analysiere alle Regionen mit erweiterter Optimierung"""
    
    results_all = {}
    
    for region in ['california', 'japan', 'chile']:
        print(f"\n{'='*70}")
        print(f"ANALYSIERE {region.upper()}")
        print('='*70)
        
        try:
            system = run_advanced_optimized_analysis(region, years_back=40)
            
            if system and hasattr(system, 'optimal_params'):
                results_all[region] = {
                    'optimal_params': system.optimal_params,
                    'bootstrap_results': system.bootstrap_results if hasattr(system, 'bootstrap_results') else None,
                    'cv_results': system.cv_results if hasattr(system, 'cv_results') else None
                }
        
        except Exception as e:
            print(f"Fehler bei {region}: {e}")
            import traceback
            traceback.print_exc()
    
    # Vergleiche Regionen
    if results_all:
        print("\n" + "="*70)
        print("REGIONALER VERGLEICH - ERWEITERTE OPTIMIERUNG")
        print("="*70)
        
        for region, results in results_all.items():
            print(f"\n{region.upper()}:")
            
            if 'optimal_params' in results:
                for mag_key, params in results['optimal_params'].items():
                    print(f"  {mag_key}:")
                    print(f"    σ_c = {params['sigma_c']:.5f}")
                    print(f"    time = {params['time_window']}d")
                    print(f"    threshold = {params['decision_threshold']:.3f}")
                    print(f"    score = {params['score']:.3f}")
            
            if 'bootstrap_results' in results and results['bootstrap_results']:
                print("\n  Performance:")
                for mag_key, metrics in results['bootstrap_results'].items():
                    if 'f1s' in metrics:
                        print(f"    {mag_key}: F1 = {metrics['f1s']['mean']:.3f}")
        
        # Speichere Gesamt-Ergebnisse
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"sizemo_all_regions_advanced_{timestamp}.json", 'w') as f:
            json.dump(results_all, f, indent=2, default=str)
        
        print(f"\nAlle Ergebnisse gespeichert: sizemo_all_regions_advanced_{timestamp}.json")
    
    return results_all


if __name__ == "__main__":
    # Einzelne Region analysieren (empfohlen für Tests)
    # system = run_advanced_optimized_analysis('california', years_back=40)
    
    # Oder alle Regionen mit erweiterter Optimierung
    results = run_all_regions_advanced()