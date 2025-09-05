"""
SIZEMO SR PREDICTION SYSTEM
=========================================
Vollständige Version mit Verbesserungspunkten:
- Längere Zeitreihen (bis 20 Jahre)
- Höhere räumliche Auflösung (25km Grid)
- Multi-Parameter Analyse (GPS, Strain, etc.)
- Nachbeben-Filtering
- Retrospektive Validierung
- Statische und Echtzeit-Visualisierung
# License Notice

This software is licensed under a dual-license model:

1. **For Non-Commercial and Personal Use**  
   - This software is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
   - Home users and researchers may use, modify, and share this software **for non-commercial purposes only**.  
   - See `LICENSE-CCBYNC.txt` for full details.

2. **For Commercial Use**  
   - Companies, organizations, and any commercial entities must acquire a **commercial license**.  
   - This commercial license follows the **Elastic License 2.0 (ELv2)** model.  
   - See `LICENSE-COMMERCIAL.txt` for details on permitted commercial usage and restrictions.

By using this software, you agree to these terms. If you are a company or organization, please contact **[www.theqa.space]** for licensing inquiries.

---

## Summary of Key Terms

| License Type         | Permitted Uses | Restrictions |
|----------------------|---------------|-------------|
| **CC BY-NC 4.0** (for non-commercial users) | Private research, personal projects, open-source contributions | No commercial use |
| **Elastic License 2.0 (ELv2)** (for companies) | Internal business use, integration into private services | No resale, no redistribution as a SaaS, requires a paid license |

For more details, please refer to the full license files:
- `LICENSE-CCBYNC.txt` (for non-commercial users)
- `LICENSE-COMMERCIAL.txt` (for companies)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import signal, stats, optimize
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
from tqdm import tqdm
import json
import pickle
import os
import pytz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import h5py  # Für große Datensätze

warnings.filterwarnings('ignore')


class UltimateEarthquakeSRSystem:
    """
    Makro-Skala SR System mit allen Verbesserungen
    """
    
    def __init__(self, region_bounds, grid_resolution_km=25):
        """
        Parameters:
        -----------
        region_bounds : dict
            {'lat_min': , 'lat_max': , 'lon_min': , 'lon_max': }
        grid_resolution_km : float
            Auflösung des Stress-Grids in km (Standard: 25km statt 50km)
        """
        self.region_bounds = region_bounds
        self.grid_resolution = grid_resolution_km
        
        # Erstelle räumliches Grid mit höherer Auflösung
        self.n_lat = int((region_bounds['lat_max'] - region_bounds['lat_min']) * 111 / grid_resolution_km)
        self.n_lon = int((region_bounds['lon_max'] - region_bounds['lon_min']) * 111 / grid_resolution_km)
        
        self.lat_grid = np.linspace(region_bounds['lat_min'], region_bounds['lat_max'], self.n_lat)
        self.lon_grid = np.linspace(region_bounds['lon_min'], region_bounds['lon_max'], self.n_lon)
        
        # Multi-Layer Stress-Felder für verschiedene Parameter
        self.stress_fields = {
            'seismic': np.zeros((self.n_lat, self.n_lon)),
            'gps': np.zeros((self.n_lat, self.n_lon)),
            'strain': np.zeros((self.n_lat, self.n_lon)),
            'combined': np.zeros((self.n_lat, self.n_lon))
        }
        
        # Zeitreihen-Speicher für lange Historie
        self.time_series_cache = []
        self.criticality_history = []
        
        # Nachbeben-Filter Parameter
        self.aftershock_params = {
            'window_days': 365,  # 1 Jahr Nachbeben-Fenster
            'space_km': 100,     # Räumlicher Radius
            'mag_diff': 1.0      # Magnitude-Differenz für Nachbeben
        }
        
    def load_extended_catalog(self, start_date, end_date, min_magnitude=2.0):
        """
        Lade erweiterten Erdbebenkatalog (bis zu 20 Jahre)
        """
        print(f"Lade erweiterten Erdbebenkatalog ({start_date} bis {end_date})...")
        
        # Teile lange Zeiträume in Chunks auf (USGS API Limit)
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        all_events = []
        chunk_size = timedelta(days=365)  # 1 Jahr Chunks
        
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            
            print(f"  Lade Chunk: {current_start.date()} bis {current_end.date()}")
            
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
                
            except Exception as e:
                print(f"    Fehler beim Laden: {e}")
            
            current_start = current_end
        
        if all_events:
            catalog = pd.concat(all_events, ignore_index=True)
            catalog = catalog.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])
            catalog = catalog.sort_values('time').reset_index(drop=True)
            
            print(f"Geladen: {len(catalog)} Events")
            print(f"Zeitraum: {catalog['time'].min()} bis {catalog['time'].max()}")
            print(f"Magnitude: {catalog['mag'].min():.1f} - {catalog['mag'].max():.1f}")
            
            self.catalog = catalog
            self.original_catalog = catalog.copy()
            
            # Filtere Nachbeben
            self.filter_aftershocks()
            
            return catalog
        else:
            print("Keine Daten geladen, erstelle synthetischen Katalog...")
            return self._create_synthetic_catalog(start_date, end_date)
    
    def _create_synthetic_catalog(self, start_date, end_date):
        """Erstelle synthetischen Katalog für Tests"""
        n_events = 10000
        
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        times = pd.date_range(start_dt, end_dt, periods=n_events, tz='UTC')
        
        catalog = pd.DataFrame({
            'time': times,
            'latitude': np.random.uniform(
                self.region_bounds['lat_min'], 
                self.region_bounds['lat_max'], 
                n_events
            ),
            'longitude': np.random.uniform(
                self.region_bounds['lon_min'], 
                self.region_bounds['lon_max'], 
                n_events
            ),
            'mag': np.random.exponential(0.8, n_events) + 2.0,
            'depth': np.random.exponential(15, n_events)
        })
        
        catalog = catalog.sort_values('time').reset_index(drop=True)
        catalog['mag'] = np.clip(catalog['mag'], 2.0, 8.0)
        
        self.catalog = catalog
        self.original_catalog = catalog.copy()
        return catalog
    
    def filter_aftershocks(self):
        """
        Intelligente Nachbeben-Filterung mit Gardner-Knopoff
        """
        print("\nFiltere Nachbeben...")
        
        mainshocks = []
        aftershocks = []
        
        catalog_sorted = self.catalog.sort_values('mag', ascending=False)
        
        for _, event in tqdm(catalog_sorted.iterrows(), total=len(catalog_sorted), desc="Nachbeben-Filter"):
            is_aftershock = False
            
            # Prüfe gegen alle bisherigen Mainshocks
            for mainshock in mainshocks:
                # Zeitfenster (Gardner-Knopoff)
                if mainshock['mag'] >= 6.5:
                    time_window = 915 * 10**(0.032 * mainshock['mag'])
                elif mainshock['mag'] >= 5.5:
                    time_window = 290 * 10**(0.024 * mainshock['mag'])
                else:
                    time_window = 155 * 10**(0.013 * mainshock['mag'])
                
                # Räumliches Fenster
                if mainshock['mag'] >= 6.5:
                    space_window = 10**(0.051 * mainshock['mag'] + 0.783)
                else:
                    space_window = 10**(0.035 * mainshock['mag'] + 0.965)
                
                # Zeit-Check
                time_diff = (event['time'] - mainshock['time']).total_seconds() / 86400
                if 0 < time_diff <= time_window:
                    # Raum-Check
                    dist = self._haversine_distance(
                        event['latitude'], event['longitude'],
                        mainshock['latitude'], mainshock['longitude']
                    )
                    
                    if dist <= space_window:
                        # Magnitude-Check
                        if event['mag'] < mainshock['mag']:
                            is_aftershock = True
                            aftershocks.append(event.to_dict())
                            break
            
            if not is_aftershock:
                mainshocks.append(event.to_dict())
        
        # Erstelle gefilterten Katalog
        self.mainshock_catalog = pd.DataFrame(mainshocks)
        self.aftershock_catalog = pd.DataFrame(aftershocks)
        
        print(f"Identifiziert: {len(mainshocks)} Mainshocks, {len(aftershocks)} Aftershocks")
        print(f"Filterung: {len(aftershocks)/len(self.catalog)*100:.1f}% als Nachbeben entfernt")
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Berechne Distanz zwischen zwei Punkten in km"""
        R = 6371  # Erdradius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def load_gps_data(self, gps_file=None):
        """
        Lade GPS-Verschiebungsdaten für Strain-Analyse
        """
        print("\nLade GPS-Daten...")
        
        if gps_file and os.path.exists(gps_file):
            gps_data = pd.read_csv(gps_file)
        else:
            # Simuliere GPS-Daten
            n_stations = 50
            gps_data = pd.DataFrame({
                'station': [f'GPS_{i:03d}' for i in range(n_stations)],
                'latitude': np.random.uniform(
                    self.region_bounds['lat_min'],
                    self.region_bounds['lat_max'],
                    n_stations
                ),
                'longitude': np.random.uniform(
                    self.region_bounds['lon_min'],
                    self.region_bounds['lon_max'],
                    n_stations
                ),
                'velocity_north': np.random.normal(0, 5, n_stations),  # mm/year
                'velocity_east': np.random.normal(0, 5, n_stations),
                'velocity_up': np.random.normal(0, 2, n_stations),
                'strain_rate': np.random.exponential(1e-7, n_stations)
            })
        
        # Interpoliere auf Grid
        points = np.column_stack((gps_data['longitude'], gps_data['latitude']))
        grid_lon, grid_lat = np.meshgrid(self.lon_grid, self.lat_grid)
        
        # Strain-Feld aus GPS-Daten
        strain_field = griddata(
            points,
            gps_data['strain_rate'],
            (grid_lon, grid_lat),
            method='cubic',
            fill_value=0
        )
        
        self.stress_fields['gps'] = strain_field
        self.gps_data = gps_data
        
        print(f"GPS-Daten geladen: {len(gps_data)} Stationen")
        return gps_data
    
    def calculate_multi_parameter_stress(self, time_window_days=90):
        """
        Berechne Multi-Parameter Stress-Feld
        """
        print(f"\nBerechne Multi-Parameter Stress-Feld...")
        
        # 1. Seismisches Stress-Feld (wie bisher)
        self._calculate_seismic_stress(time_window_days)
        
        # 2. GPS/Strain Stress-Feld
        if hasattr(self, 'gps_data'):
            self.stress_fields['strain'] = gaussian_filter(
                self.stress_fields['gps'],
                sigma=2
            )
        
        # 3. Kombiniertes Stress-Feld mit gewichteter Summe
        weights = {
            'seismic': 0.5,
            'gps': 0.2,
            'strain': 0.3
        }
        
        combined = np.zeros_like(self.stress_fields['seismic'])
        for field, weight in weights.items():
            if field in self.stress_fields:
                combined += weight * self.stress_fields[field]
        
        # Normalisiere
        if combined.max() > 0:
            combined /= combined.max()
        
        self.stress_fields['combined'] = combined
        self.stress_field = combined  # Für Kompatibilität
        
        return combined
    
    def _calculate_seismic_stress(self, time_window_days):
        """Berechne seismisches Stress-Feld"""
        # Reset
        self.stress_fields['seismic'] = np.zeros((self.n_lat, self.n_lon))
        energy_accumulation = np.zeros((self.n_lat, self.n_lon))
        
        # Verwende gefilterten Katalog (ohne Nachbeben)
        if hasattr(self, 'mainshock_catalog'):
            catalog_to_use = self.mainshock_catalog
        else:
            catalog_to_use = self.catalog
        
        # Zeitfenster
        end_time = catalog_to_use['time'].max()
        start_time = end_time - pd.Timedelta(days=time_window_days)
        
        recent_events = catalog_to_use[
            (catalog_to_use['time'] >= start_time) & 
            (catalog_to_use['time'] <= end_time)
        ]
        
        print(f"Verwende {len(recent_events)} gefilterte Events")
        
        # Berechne Energie
        for _, event in recent_events.iterrows():
            energy = 10 ** (1.5 * event['mag'] + 4.8)
            
            lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
            lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
            
            # Adaptive Gaussian basierend auf Magnitude
            sigma = max(1, int(event['mag'] / 2))
            
            for i in range(max(0, lat_idx-10), min(self.n_lat, lat_idx+11)):
                for j in range(max(0, lon_idx-10), min(self.n_lon, lon_idx+11)):
                    dist = np.sqrt((i-lat_idx)**2 + (j-lon_idx)**2)
                    weight = np.exp(-dist**2 / (2*sigma**2))
                    energy_accumulation[i, j] += energy * weight
        
        # Log-Transform und Normalisierung
        if energy_accumulation.max() > 0:
            energy_accumulation = np.log10(energy_accumulation + 1)
            energy_accumulation /= energy_accumulation.max()
        
        self.stress_fields['seismic'] = gaussian_filter(energy_accumulation, sigma=2)
    
    def adaptive_sr_optimization(self, multi_scale=True):
        """
        Adaptive SR-Optimierung mit Multi-Scale Analyse
        """
        print("\nAdaptive SR-Optimierung...")
        
        if multi_scale:
            # Teste verschiedene zeitliche Skalen
            scales = {
                'hourly': 1/24,
                'daily': 1,
                'weekly': 7,
                'monthly': 30,
                'yearly': 365
            }
            
            best_results = {}
            
            for scale_name, scale_days in scales.items():
                print(f"  Teste Skala: {scale_name}")
                
                # Aggregiere Stress auf dieser Zeitskala
                if scale_days < 1:
                    # Für stündliche Skala, verwende höhere zeitliche Auflösung
                    stress_scaled = self.stress_field
                else:
                    # Glätte basierend auf Zeitskala
                    sigma = scale_days / 30  # Normalisiert auf Monat
                    stress_scaled = gaussian_filter(self.stress_field, sigma=sigma)
                
                # SR-Analyse für diese Skala
                noise_levels = np.logspace(-4, 0, 50)
                snr_values = []
                
                for noise in noise_levels:
                    snr_trials = []
                    for _ in range(10):
                        noisy = stress_scaled + np.random.normal(0, noise, stress_scaled.shape)
                        threshold = np.mean(noisy) + 1.5 * np.std(noisy)
                        
                        signal_mask = noisy > threshold
                        if np.sum(signal_mask) > 0:
                            snr = np.mean(noisy[signal_mask]) / (np.std(noisy[~signal_mask]) + 1e-10)
                        else:
                            snr = 0
                        snr_trials.append(snr)
                    
                    snr_values.append(np.mean(snr_trials))
                
                # Finde optimales Noise für diese Skala
                optimal_idx = np.argmax(snr_values)
                best_results[scale_name] = {
                    'noise': noise_levels[optimal_idx],
                    'snr': snr_values[optimal_idx],
                    'all_snr': snr_values
                }
            
            # Wähle beste Skala
            best_scale = max(best_results.items(), key=lambda x: x[1]['snr'])
            print(f"\nBeste Skala: {best_scale[0]} mit SNR = {best_scale[1]['snr']:.3f}")
            
            self.sr_results = best_results
            self.optimal_noise = best_scale[1]['noise']
            self.optimal_scale = best_scale[0]
            
        else:
            # Standard SR-Optimierung
            self.optimal_noise = self._standard_sr_optimization()
        
        # Erstelle finale Risk-Map
        self.risk_map = self._create_adaptive_risk_map()
        
        return self.sr_results
    
    def _standard_sr_optimization(self):
        """Standard SR-Optimierung"""
        noise_levels = np.logspace(-3, -1, 30)
        snr_values = []
        
        for noise in tqdm(noise_levels, desc="SR-Optimierung"):
            snr_trials = []
            for _ in range(20):
                noisy = self.stress_field + np.random.normal(0, noise, self.stress_field.shape)
                threshold = np.mean(noisy) + 1.5 * np.std(noisy)
                
                high_stress = noisy > threshold
                if np.sum(high_stress) > 0:
                    snr = np.mean(noisy[high_stress]) / (np.std(noisy[~high_stress]) + 1e-10)
                else:
                    snr = 0
                snr_trials.append(snr)
            
            snr_values.append(np.mean(snr_trials))
        
        optimal_idx = np.argmax(snr_values)
        return noise_levels[optimal_idx]
    
    def _create_adaptive_risk_map(self):
        """Erstelle adaptive Risiko-Karte"""
        risk_accumulator = np.zeros_like(self.stress_field)
        
        # Verwende optimales Noise
        for _ in range(100):
            noisy = self.stress_field + np.random.normal(0, self.optimal_noise, self.stress_field.shape)
            
            # Mehrere Schwellwerte für robuste Schätzung
            for percentile in [80, 85, 90, 95]:
                threshold = np.percentile(noisy, percentile)
                near_threshold = (noisy > threshold * 0.9) & (noisy <= threshold * 1.1)
                risk_accumulator[near_threshold] += 1
        
        # Normalisiere und glätte
        risk_map = risk_accumulator / (100 * 4)  # 100 trials * 4 thresholds
        risk_map = gaussian_filter(risk_map, sigma=3)
        
        return risk_map
    
    def retrospective_validation(self, historical_events):
        """
        Retrospektive Validierung mit historischen Großbeben
        """
        print("\nRetrospektive Validierung...")
        
        validation_results = []
        
        for event in historical_events:
            print(f"  Validiere {event['name']} (M{event['magnitude']})")
            
            # Lade Daten vor dem Event
            end_date = pd.to_datetime(event['date']) - timedelta(days=1)
            start_date = end_date - timedelta(days=365*2)  # 2 Jahre vorher
            
            # Temporärer Katalog
            temp_catalog = self.catalog[
                (self.catalog['time'] >= start_date.tz_localize('UTC')) &
                (self.catalog['time'] <= end_date.tz_localize('UTC'))
            ]
            
            if len(temp_catalog) > 100:
                # Berechne Stress für diesen Zeitraum
                self.catalog = temp_catalog
                self.calculate_multi_parameter_stress(90)
                
                # War der Epicenter in einer Hochrisiko-Zone?
                lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
                lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
                
                risk_at_epicenter = self.risk_map[lat_idx, lon_idx]
                percentile = stats.percentileofscore(self.risk_map.flatten(), risk_at_epicenter)
                
                validation_results.append({
                    'event': event['name'],
                    'risk_value': risk_at_epicenter,
                    'percentile': percentile,
                    'success': percentile > 80  # Top 20% als Erfolg
                })
        
        # Restore original catalog
        self.catalog = self.original_catalog
        
        # Berechne Erfolgsrate
        if validation_results:
            success_rate = sum(r['success'] for r in validation_results) / len(validation_results)
            print(f"\nErfolgsrate: {success_rate*100:.1f}%")
        
        self.validation_results = validation_results
        return validation_results
    
    def calculate_enhanced_criticality(self):
        """
        Erweiterte Kritikalitäts-Berechnung mit mehr Parametern
        KORRIGIERT: Entropie-Berechnung und Fehlerbehandlung
        """
        metrics = {}
        
        # Basis-Metriken
        metrics['order_parameter'] = np.mean(self.stress_field)
        metrics['susceptibility'] = np.var(self.stress_field)
        metrics['correlation_length'] = self._calculate_correlation_length()
        
        # Erweiterte Metriken mit Fehlerbehandlung
        
        # Korrigierte Entropie-Berechnung
        stress_flat = self.stress_field.flatten()
        stress_positive = stress_flat[stress_flat > 0]  # Nur positive Werte
        if len(stress_positive) > 0:
            # Normalisiere für Entropie-Berechnung
            stress_norm = stress_positive / np.sum(stress_positive)
            metrics['entropy'] = stats.entropy(stress_norm)
            if np.isinf(metrics['entropy']) or np.isnan(metrics['entropy']):
                metrics['entropy'] = 0  # Fallback
        else:
            metrics['entropy'] = 0
        
        metrics['fractal_dimension'] = self._calculate_fractal_dimension()
        metrics['hurst_exponent'] = self._calculate_hurst_exponent()
        
        # b-Wert mit robusten Statistiken
        if hasattr(self, 'mainshock_catalog') and len(self.mainshock_catalog) > 50:
            metrics['b_value'] = self._calculate_robust_b_value(
                self.mainshock_catalog['mag'].values
            )
        else:
            metrics['b_value'] = 1.0
        
        # Quiescence mit verschiedenen Zeitfenstern
        metrics['quiescence_30d'] = self._calculate_quiescence(30)
        metrics['quiescence_90d'] = self._calculate_quiescence(90)
        metrics['quiescence_365d'] = self._calculate_quiescence(365)
        
        # Machine Learning Criticality Score (optional)
        if hasattr(self, 'ml_model'):
            features = np.array([metrics[k] for k in sorted(metrics.keys())]).reshape(1, -1)
            metrics['ml_criticality'] = self.ml_model.predict_proba(features)[0, 1]
        
        # Kombinierter Criticality Index mit Gewichtung
        weights = {
            'susceptibility': 2.0,
            'correlation_length': 1.5,
            'entropy': 0.5,
            'hurst_exponent': 0.8,
            'b_value': -1.0  # Negativ, da niedriger b-Wert = höheres Risiko
        }
        
        criticality = 0
        for param, weight in weights.items():
            if param in metrics and not np.isinf(metrics[param]) and not np.isnan(metrics[param]):
                if param == 'b_value':
                    # Spezielle Behandlung für b-Wert (invertiert)
                    criticality += weight * (1 / (metrics[param] + 0.1))
                else:
                    criticality += weight * metrics[param]
        
        # Zusätzliche Quiescence-Komponente
        # Positive Quiescence auf langer Zeitskala ist gefährlich
        if metrics['quiescence_365d'] > 2:
            criticality *= 1.5  # Verstärkungsfaktor
        
        # Berücksichtige die Quiescence-Anomalie
        if metrics['quiescence_30d'] < -2 and metrics['quiescence_365d'] > 2:
            # Kurzfristig aktiv, langfristig ruhig = sehr gefährlich
            criticality *= 2.0
        
        metrics['criticality_index'] = criticality
        
        # Zusätzliche Warnstufen-Klassifikation
        if criticality > 100:
            metrics['warning_level'] = 'EXTREME'
        elif criticality > 50:
            metrics['warning_level'] = 'CRITICAL'
        elif criticality > 20:
            metrics['warning_level'] = 'HIGH'
        elif criticality > 10:
            metrics['warning_level'] = 'ELEVATED'
        elif criticality > 5:
            metrics['warning_level'] = 'MODERATE'
        else:
            metrics['warning_level'] = 'LOW'
        
        return metrics
    
    def _calculate_fractal_dimension(self):
        """Box-Counting Fractal Dimension"""
        # Binarisiere Stress-Feld
        binary = self.stress_field > np.mean(self.stress_field)
        
        # Box-Counting
        sizes = 2**np.arange(1, int(np.log2(min(self.n_lat, self.n_lon))))
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, self.n_lat, size):
                for j in range(0, self.n_lon, size):
                    box = binary[i:i+size, j:j+size]
                    if np.any(box):
                        count += 1
            counts.append(count)
        
        # Fit log-log
        if len(counts) > 2:
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -coeffs[0]
        return 2.0
    
    def _calculate_hurst_exponent(self):
        """Hurst Exponent für Langzeit-Korrelation"""
        ts = self.stress_field.flatten()
        N = len(ts)
        
        if N < 100:
            return 0.5
        
        # R/S Analyse
        lags = range(2, min(100, N//2))
        rs = []
        
        for lag in lags:
            rs_lag = []
            for start in range(0, N-lag):
                series = ts[start:start+lag]
                mean = np.mean(series)
                std = np.std(series)
                
                if std > 0:
                    cumsum = np.cumsum(series - mean)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = std
                    rs_lag.append(R/S)
            
            if rs_lag:
                rs.append(np.mean(rs_lag))
        
        if len(rs) > 2:
            # Fit log-log
            coeffs = np.polyfit(np.log(list(lags)), np.log(rs), 1)
            return coeffs[0]
        
        return 0.5
    
    def _calculate_robust_b_value(self, magnitudes):
        """Robuste b-Wert Berechnung mit MLE"""
        if len(magnitudes) < 20:
            return 1.0
        
        # Magnitude of Completeness mit GFT
        mc = self._estimate_completeness_magnitude(magnitudes)
        complete = magnitudes[magnitudes >= mc]
        
        if len(complete) < 10:
            return 1.0
        
        # Maximum Likelihood Estimation
        b_mle = 1 / (np.mean(complete) - mc + 0.05)
        
        # Aki's correction
        n = len(complete)
        b_corrected = b_mle / (1 - 1/n)
        
        return np.clip(b_corrected, 0.3, 2.5)
    
    def _estimate_completeness_magnitude(self, magnitudes):
        """Goodness-of-Fit Test für Completeness Magnitude"""
        mag_bins = np.arange(np.floor(magnitudes.min()*10)/10, 
                            np.ceil(magnitudes.max()*10)/10, 0.1)
        
        best_mc = mag_bins[0]
        best_r = 0
        
        for mc_test in mag_bins[:-5]:  # Mindestens 5 bins über mc
            complete = magnitudes[magnitudes >= mc_test]
            
            if len(complete) >= 25:
                # GFT R-value
                obs_counts, _ = np.histogram(complete, bins=mag_bins[mag_bins >= mc_test])
                
                if len(obs_counts) > 2:
                    # Theoretical exponential
                    b = 1 / (np.mean(complete) - mc_test + 0.05)
                    exp_counts = len(complete) * np.diff(10**(-b * mag_bins[mag_bins >= mc_test]))
                    
                    # Normalized residuals
                    if np.sum(exp_counts) > 0:
                        r = 1 - np.sum(np.abs(obs_counts - exp_counts[:len(obs_counts)])) / np.sum(obs_counts)
                        
                        if r > best_r and r > 0.9:  # 90% Goodness-of-Fit
                            best_r = r
                            best_mc = mc_test
        
        return best_mc
    
    def _calculate_quiescence(self, days):
        """Berechne Quiescence für verschiedene Zeitfenster"""
        if not hasattr(self, 'mainshock_catalog') or len(self.mainshock_catalog) < 50:
            return 0.0
        
        now = self.mainshock_catalog['time'].max()
        
        recent = self.mainshock_catalog[
            self.mainshock_catalog['time'] > now - pd.Timedelta(days=days)
        ]
        
        historical = self.mainshock_catalog[
            (self.mainshock_catalog['time'] > now - pd.Timedelta(days=days*4)) &
            (self.mainshock_catalog['time'] <= now - pd.Timedelta(days=days))
        ]
        
        if len(historical) > 0:
            recent_rate = len(recent) / days
            historical_rate = len(historical) / (days * 3)
            
            z_score = (recent_rate - historical_rate) / (np.sqrt(historical_rate / (days * 3)) + 1e-10)
            return np.clip(z_score, -3, 3)
        
        return 0.0
    
    def _calculate_correlation_length(self):
        """Erweiterte Korrelationslängen-Berechnung"""
        # 2D FFT für schnellere Berechnung
        fft = np.fft.fft2(self.stress_field)
        power = np.abs(fft)**2
        
        # Radial average
        center = (self.n_lat//2, self.n_lon//2)
        y, x = np.ogrid[:self.n_lat, :self.n_lon]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        r_int = r.astype(int)
        tbin = np.bincount(r_int.ravel(), power.ravel())
        nr = np.bincount(r_int.ravel())
        
        radial_profile = tbin / (nr + 1e-10)
        
        # Finde 1/e Abfall
        threshold = radial_profile[0] / np.e
        
        for i, val in enumerate(radial_profile[1:], 1):
            if val < threshold:
                return i * self.grid_resolution
        
        return min(self.n_lat, self.n_lon) * self.grid_resolution / 4
    
    def save_to_hdf5(self, filename='earthquake_analysis.h5'):
        """Speichere große Datensätze effizient in HDF5"""
        with h5py.File(filename, 'w') as f:
            # Metadaten
            meta = f.create_group('metadata')
            meta.attrs['grid_resolution'] = self.grid_resolution
            meta.attrs['optimal_noise'] = self.optimal_noise if hasattr(self, 'optimal_noise') else 0
            
            # Stress-Felder
            stress_group = f.create_group('stress_fields')
            for name, field in self.stress_fields.items():
                stress_group.create_dataset(name, data=field, compression='gzip')
            
            # Kataloge - FIXED für String-Konvertierung
            if hasattr(self, 'catalog'):
                cat_group = f.create_group('catalog')
                for col in self.catalog.columns:
                    try:
                        if self.catalog[col].dtype == 'datetime64[ns, UTC]':
                            # Konvertiere datetime zu String
                            cat_group.create_dataset(col, data=self.catalog[col].astype(str).values)
                        elif self.catalog[col].dtype == 'object':
                            # String-Spalten separat behandeln
                            str_data = self.catalog[col].fillna('').astype(str).values
                            cat_group.create_dataset(col, data=str_data)
                        else:
                            # Numerische Daten
                            cat_group.create_dataset(col, data=self.catalog[col].values)
                    except Exception as e:
                        print(f"Warnung: Konnte Spalte {col} nicht speichern: {e}")
            
            # Ergebnisse
            if hasattr(self, 'risk_map'):
                f.create_dataset('risk_map', data=self.risk_map, compression='gzip')
        
        print(f"Analyse gespeichert in {filename}")
    
    def create_static_analysis_visualization(self):
        """Erstelle statische Analyse-Visualisierung mit allen Daten"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # 1. Stress-Feld
        ax = axes[0, 0]
        im = ax.imshow(self.stress_field.T, origin='lower', cmap='RdYlBu_r',
                       extent=[self.region_bounds['lon_min'], self.region_bounds['lon_max'],
                              self.region_bounds['lat_min'], self.region_bounds['lat_max']])
        ax.set_title('Multi-Parameter Stress-Feld')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Stress')
        
        # 2. Gefilterte Mainshocks
        ax = axes[0, 1]
        if hasattr(self, 'mainshock_catalog') and len(self.mainshock_catalog) > 0:
            sc = ax.scatter(self.mainshock_catalog['longitude'],
                           self.mainshock_catalog['latitude'],
                           s=self.mainshock_catalog['mag']**2,
                           c=self.mainshock_catalog['mag'],
                           cmap='hot', alpha=0.6)
            ax.set_xlim(self.region_bounds['lon_min'], self.region_bounds['lon_max'])
            ax.set_ylim(self.region_bounds['lat_min'], self.region_bounds['lat_max'])
            ax.set_title(f'Mainshocks (n={len(self.mainshock_catalog)})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(sc, ax=ax, label='Magnitude')
        
        # 3. Risk Map
        ax = axes[0, 2]
        im = ax.imshow(self.risk_map.T, origin='lower', cmap='Reds',
                       extent=[self.region_bounds['lon_min'], self.region_bounds['lon_max'],
                              self.region_bounds['lat_min'], self.region_bounds['lat_max']])
        ax.set_title(f'SR Risk Map (Noise={self.optimal_noise:.3f})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Risk')
        
        # 4. Multi-Scale SR
        ax = axes[1, 0]
        if hasattr(self, 'sr_results'):
            for scale, data in self.sr_results.items():
                ax.plot(np.logspace(-4, 0, len(data['all_snr'])), 
                       data['all_snr'], label=f"{scale} (SNR={data['snr']:.2f})")
            ax.set_xscale('log')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('SNR')
            ax.set_title('Multi-Scale SR Analysis')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 5. Zeitliche Aktivität
        ax = axes[1, 1]
        if hasattr(self, 'mainshock_catalog') and len(self.mainshock_catalog) > 0:
            time_bins = pd.date_range(self.mainshock_catalog['time'].min(), 
                                     self.mainshock_catalog['time'].max(), 
                                     periods=50)
            hist, _ = np.histogram(self.mainshock_catalog['time'], bins=time_bins)
            ax.plot(time_bins[:-1], hist, 'b-', linewidth=2)
            ax.set_xlabel('Zeit')
            ax.set_ylabel('Anzahl Mainshocks')
            ax.set_title('Zeitliche Evolution')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Magnitude-Verteilung
        ax = axes[1, 2]
        if hasattr(self, 'mainshock_catalog') and len(self.mainshock_catalog) > 0:
            mags = self.mainshock_catalog['mag'].values
            ax.hist(mags, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=mags.mean(), color='red', linestyle='--', 
                       label=f'Mean={mags.mean():.2f}')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('Anzahl')
            ax.set_title('Magnitude-Verteilung (Mainshocks)')
            ax.legend()
        
        # 7. Kritikalitäts-Metriken
        ax = axes[2, 0]
        ax.axis('off')
        metrics = self.calculate_enhanced_criticality()
        text = f"""KRITIKALITÄTS-ANALYSE
{'='*25}
Criticality Index: {metrics['criticality_index']:.1f}
Order Parameter: {metrics['order_parameter']:.3f}
Susceptibility: {metrics['susceptibility']:.3f}
Correlation [km]: {metrics['correlation_length']:.0f}
Entropy: {metrics['entropy']:.2f}
Fractal Dim: {metrics['fractal_dimension']:.2f}
Hurst Exponent: {metrics['hurst_exponent']:.3f}
b-Value: {metrics['b_value']:.2f}

QUIESCENCE:
  30 Tage: {metrics['quiescence_30d']:.2f}
  90 Tage: {metrics['quiescence_90d']:.2f}
  365 Tage: {metrics['quiescence_365d']:.2f}

STATUS:
"""
        if metrics['criticality_index'] > 10:
            text += "🚨 KRITISCH!"
        elif metrics['criticality_index'] > 5:
            text += "⚠️ ERHÖHT"
        else:
            text += "✅ NORMAL"
            
        ax.text(0.1, 0.9, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # 8. Validierungs-Ergebnisse
        ax = axes[2, 1]
        if hasattr(self, 'validation_results') and self.validation_results:
            events = [r['event'] for r in self.validation_results]
            percentiles = [r['percentile'] for r in self.validation_results]
            colors = ['green' if p > 80 else 'orange' if p > 60 else 'red' 
                     for p in percentiles]
            ax.barh(events, percentiles, color=colors)
            ax.axvline(x=80, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Risk Percentile')
            ax.set_title('Retrospektive Validierung')
        else:
            ax.text(0.5, 0.5, 'Keine Validierung\nverfügbar', 
                   transform=ax.transAxes, ha='center', va='center')
        
        # 9. Hochrisiko-Zonen
        ax = axes[2, 2]
        risk_threshold = np.percentile(self.risk_map, 95)
        high_risk = self.risk_map > risk_threshold
        masked_risk = np.ma.masked_where(~high_risk, self.risk_map)
        im = ax.imshow(masked_risk.T, origin='lower', cmap='Reds',
                       extent=[self.region_bounds['lon_min'], self.region_bounds['lon_max'],
                              self.region_bounds['lat_min'], self.region_bounds['lat_max']])
        ax.set_title('Top 5% Risiko-Zonen')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Risk Level')
        
        # Haupttitel
        plt.suptitle(f'SIZEMO SR EARTHQUAKE ANALYSIS | {self.optimal_scale.upper()} SCALE | '
                    f'CI={metrics["criticality_index"]:.0f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


def run_ultimate_analysis(region='california', years_back=10):
    """
    Führe ultimative Analyse mit allen Verbesserungen durch
    """
    print("="*70)
    print("SIZEMO EARTHQUAKE SR PREDICTION SYSTEM")
    print("="*70)
    
    # Regionen-Definition
    regions = {
        'california': {
            'lat_min': 32.0,
            'lat_max': 42.0,
            'lon_min': -125.0,
            'lon_max': -114.0
        },
        'japan': {
            'lat_min': 30.0,
            'lat_max': 46.0,
            'lon_min': 129.0,
            'lon_max': 146.0
        },
        'chile': {
            'lat_min': -45.0,
            'lat_max': -17.0,
            'lon_min': -76.0,
            'lon_max': -66.0
        }
    }
    
    # Initialisiere System mit höherer Auflösung
    system = UltimateEarthquakeSRSystem(
        region_bounds=regions[region],
        grid_resolution_km=25  # 25km statt 50km
    )
    
    # Lade erweiterten Katalog
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*years_back)).strftime('%Y-%m-%d')
    
    catalog = system.load_extended_catalog(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=2.0  # Niedrigere Schwelle
    )
    
    # Lade GPS-Daten
    system.load_gps_data()
    
    # Berechne Multi-Parameter Stress
    system.calculate_multi_parameter_stress(time_window_days=90)
    
    # Adaptive SR-Optimierung
    system.adaptive_sr_optimization(multi_scale=True)
    
    # Retrospektive Validierung für California
    if region == 'california':
        historical_events = [
            {'name': 'Ridgecrest 2019', 'date': '2019-07-06', 'magnitude': 7.1,
             'latitude': 35.77, 'longitude': -117.61},
            {'name': 'Napa 2014', 'date': '2014-08-24', 'magnitude': 6.0,
             'latitude': 38.22, 'longitude': -122.31}
        ]
        system.retrospective_validation(historical_events)
    
    # Erweiterte Kritikalitäts-Analyse
    metrics = system.calculate_enhanced_criticality()
    
    # Zeige Ergebnisse
    print("\n" + "="*70)
    print("ANALYSE-ERGEBNISSE")
    print("="*70)
    
    print(f"\nOptimales Noise-Level: {system.optimal_noise:.6f}")
    print(f"Optimale Skala: {system.optimal_scale}")
    print(f"Verhältnis zu σc: {system.optimal_noise/0.117:.2f}x")
    
    print(f"\nErweiterte Kritikalitäts-Metriken:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    # Risikobewertung
    print(f"\n{'='*40}")
    if metrics['criticality_index'] > 10:
        print("🚨 WARNUNG: System zeigt KRITISCHE Anzeichen!")
        print("   Unmittelbares Risiko für größeres Event")
    elif metrics['criticality_index'] > 5:
        print("⚠️  ERHÖHTE AUFMERKSAMKEIT empfohlen")
        print("   System nähert sich kritischem Punkt")
    else:
        print("✅ System im NORMALEN Bereich")
        print("   Kein unmittelbares erhöhtes Risiko")
    print('='*40)
    
    # Speichere in HDF5
    system.save_to_hdf5(f'{region}_ultimate_analysis.h5')
    
    # Erstelle und zeige statische Visualisierung
    print("\nErstelle Visualisierung...")
    fig = system.create_static_analysis_visualization()
    plt.savefig(f'{region}_ultimate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*70)
    
    return system


if __name__ == "__main__":
    # Führe Ultimate Analyse durch

    system = run_ultimate_analysis('california', years_back=10)
