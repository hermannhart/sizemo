"""
Makro-Skala Stochastic Resonance System für Erdbebenvorhersage
===============================================================
Implementiert die "Kochende Herdplatte" Analogie mit echten seismischen Daten
KORRIGIERT: Timezone-aware datetime Vergleiche
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import signal, stats
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
from tqdm import tqdm
import json
import pickle
import os
import pytz  # Für Zeitzonenhandling

warnings.filterwarnings('ignore')


class MacroScaleSRSystem:
    """
    Makro-Skala SR System für regionale Erdbebenvorhersage
    Arbeitet auf Monats/Jahre-Skala statt Millisekunden
    """
    
    def __init__(self, region_bounds, grid_resolution_km=50):
        """
        Parameters:
        -----------
        region_bounds : dict
            {'lat_min': , 'lat_max': , 'lon_min': , 'lon_max': }
        grid_resolution_km : float
            Auflösung des Stress-Grids in km
        """
        self.region_bounds = region_bounds
        self.grid_resolution = grid_resolution_km
        
        # Erstelle räumliches Grid
        self.n_lat = int((region_bounds['lat_max'] - region_bounds['lat_min']) * 111 / grid_resolution_km)
        self.n_lon = int((region_bounds['lon_max'] - region_bounds['lon_min']) * 111 / grid_resolution_km)
        
        self.lat_grid = np.linspace(region_bounds['lat_min'], region_bounds['lat_max'], self.n_lat)
        self.lon_grid = np.linspace(region_bounds['lon_min'], region_bounds['lon_max'], self.n_lon)
        
        # Stress-Feld (das ist unsere "Herdplatte")
        self.stress_field = np.zeros((self.n_lat, self.n_lon))
        self.energy_accumulation = np.zeros((self.n_lat, self.n_lon))
        
        # Zeitreihen-Speicher
        self.time_series = []
        self.criticality_history = []
        
    def load_earthquake_catalog(self, start_date, end_date, min_magnitude=2.5):
        """
        Lade echten Erdbebenkatalog von USGS
        """
        print(f"Lade Erdbebendaten für Region...")
        
        base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        params = {
            'format': 'csv',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'minlatitude': self.region_bounds['lat_min'],
            'maxlatitude': self.region_bounds['lat_max'],
            'minlongitude': self.region_bounds['lon_min'],
            'maxlongitude': self.region_bounds['lon_max'],
            'orderby': 'time'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            catalog = pd.read_csv(StringIO(response.text))
            catalog['time'] = pd.to_datetime(catalog['time'])
            
            # WICHTIG: Stelle sicher, dass alle Zeiten UTC-aware sind
            if catalog['time'].dt.tz is None:
                catalog['time'] = catalog['time'].dt.tz_localize('UTC')
            else:
                catalog['time'] = catalog['time'].dt.tz_convert('UTC')
            
            print(f"Geladen: {len(catalog)} Events von {catalog['time'].min()} bis {catalog['time'].max()}")
            print(f"Magnitude-Bereich: {catalog['mag'].min():.1f} - {catalog['mag'].max():.1f}")
            
            self.catalog = catalog
            self.original_catalog = catalog.copy()  # Backup für temporal_evolution
            return catalog
            
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            print("Erstelle synthetischen Katalog...")
            return self._create_synthetic_catalog(start_date, end_date)
    
    def _create_synthetic_catalog(self, start_date, end_date):
        """
        Fallback: Synthetischer Katalog für Tests
        """
        n_events = 10000
        
        # Konvertiere zu UTC-aware timestamps
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
            'mag': np.random.exponential(0.8, n_events) + 2.5,
            'depth': np.random.exponential(15, n_events)
        })
        
        # Füge einige "Cluster" hinzu
        for _ in range(5):
            cluster_lat = np.random.uniform(
                self.region_bounds['lat_min'], 
                self.region_bounds['lat_max']
            )
            cluster_lon = np.random.uniform(
                self.region_bounds['lon_min'], 
                self.region_bounds['lon_max']
            )
            
            n_cluster = np.random.randint(50, 200)
            cluster_events = pd.DataFrame({
                'time': pd.date_range(
                    start_dt, 
                    end_dt, 
                    periods=n_cluster,
                    tz='UTC'
                ),
                'latitude': np.random.normal(cluster_lat, 0.5, n_cluster),
                'longitude': np.random.normal(cluster_lon, 0.5, n_cluster),
                'mag': np.random.exponential(0.5, n_cluster) + 3.0,
                'depth': np.random.exponential(10, n_cluster)
            })
            
            catalog = pd.concat([catalog, cluster_events])
        
        catalog = catalog.sort_values('time').reset_index(drop=True)
        catalog['mag'] = np.clip(catalog['mag'], 2.5, 8.0)
        
        self.catalog = catalog
        self.original_catalog = catalog.copy()
        return catalog
    
    def calculate_stress_field(self, time_window_days=30):
        """
        Berechne Stress-Feld aus seismischer Aktivität
        """
        print(f"\nBerechne Stress-Feld (Zeitfenster: {time_window_days} Tage)...")
        
        # Reset
        self.stress_field = np.zeros((self.n_lat, self.n_lon))
        self.energy_accumulation = np.zeros((self.n_lat, self.n_lon))
        
        # Zeitfenster mit UTC-aware datetime
        end_time = self.catalog['time'].max()
        start_time = end_time - pd.Timedelta(days=time_window_days)
        
        recent_events = self.catalog[
            (self.catalog['time'] >= start_time) & 
            (self.catalog['time'] <= end_time)
        ]
        
        print(f"Verwende {len(recent_events)} Events aus den letzten {time_window_days} Tagen")
        
        # Berechne Energie für jedes Event und verteile auf Grid
        for _, event in tqdm(recent_events.iterrows(), total=len(recent_events), desc="Berechne Stress"):
            # Seismische Energie
            energy = 10 ** (1.5 * event['mag'] + 4.8)
            
            # Finde nächste Grid-Zelle
            lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
            lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
            
            # Verteile Energie mit Gaussian (simuliert Stress-Transfer)
            for i in range(max(0, lat_idx-5), min(self.n_lat, lat_idx+6)):
                for j in range(max(0, lon_idx-5), min(self.n_lon, lon_idx+6)):
                    dist = np.sqrt((i-lat_idx)**2 + (j-lon_idx)**2)
                    weight = np.exp(-dist**2 / 4)  # Gaussian mit σ=2 Grid-Zellen
                    
                    self.energy_accumulation[i, j] += energy * weight
        
        # Normalisiere und glätte
        if self.energy_accumulation.max() > 0:
            self.energy_accumulation = np.log10(self.energy_accumulation + 1)
            self.energy_accumulation /= self.energy_accumulation.max()
        
        # Stress ist akkumulierte Energie + zeitliche Änderung
        self.stress_field = gaussian_filter(self.energy_accumulation, sigma=2)
        
        # Füge zeitliche Komponente hinzu (Stress baut sich auf wo lange nichts war)
        quiet_zones = self._identify_quiet_zones(recent_events)
        self.stress_field += quiet_zones * 0.3
        
        return self.stress_field
    
    def _identify_quiet_zones(self, recent_events):
        """
        Identifiziere Zonen mit wenig Aktivität (potentiell gefährlich)
        """
        activity_map = np.zeros((self.n_lat, self.n_lon))
        
        # Zähle Events pro Grid-Zelle
        for _, event in recent_events.iterrows():
            lat_idx = np.argmin(np.abs(self.lat_grid - event['latitude']))
            lon_idx = np.argmin(np.abs(self.lon_grid - event['longitude']))
            
            if 0 <= lat_idx < self.n_lat and 0 <= lon_idx < self.n_lon:
                activity_map[lat_idx, lon_idx] += 1
        
        # Invertiere und normalisiere (quiet zones haben hohe Werte)
        quiet_zones = 1 / (activity_map + 1)
        quiet_zones = gaussian_filter(quiet_zones, sigma=3)
        
        return quiet_zones
    
    def apply_stochastic_resonance(self, noise_levels=None):
        """
        Wende SR auf das Stress-Feld an
        """
        print("\nWende Stochastic Resonance an...")
        
        if noise_levels is None:
            # Teste verschiedene Noise-Level
            noise_levels = np.logspace(-3, -1, 30)  # 0.001 bis 0.1
        
        sr_results = []
        
        for noise in tqdm(noise_levels, desc="Teste Noise-Level"):
            # Mehrere Trials für Statistik
            snr_values = []
            
            for trial in range(20):
                # Füge Noise hinzu
                noisy_field = self.stress_field + np.random.normal(0, noise, self.stress_field.shape)
                
                # Schwellwert-basierte Detektion
                threshold = np.mean(noisy_field) + 1.5 * np.std(noisy_field)
                
                # Bereiche über Schwellwert
                high_stress = noisy_field > threshold
                
                # SNR Berechnung
                if np.sum(high_stress) > 0:
                    signal = np.mean(noisy_field[high_stress])
                    noise_level = np.std(noisy_field[~high_stress])
                    
                    if noise_level > 0:
                        snr = signal / noise_level
                    else:
                        snr = 0
                else:
                    snr = 0
                
                snr_values.append(snr)
            
            sr_results.append({
                'noise': noise,
                'mean_snr': np.mean(snr_values),
                'std_snr': np.std(snr_values)
            })
        
        # Finde optimales Noise
        snr_array = np.array([r['mean_snr'] for r in sr_results])
        optimal_idx = np.argmax(snr_array)
        self.optimal_noise = sr_results[optimal_idx]['noise']
        self.sr_results = sr_results
        
        print(f"Optimales Noise-Level: {self.optimal_noise:.6f}")
        print(f"Maximales SNR: {sr_results[optimal_idx]['mean_snr']:.3f}")
        
        # Erstelle finale Risk-Map mit optimalem Noise
        self.risk_map = self._create_risk_map(self.optimal_noise)
        
        return sr_results
    
    def _create_risk_map(self, noise_level):
        """
        Erstelle Risiko-Karte mit optimalem Noise
        """
        # Mehrere Realisierungen für robuste Schätzung
        risk_accumulator = np.zeros_like(self.stress_field)
        
        for _ in range(50):
            noisy_field = self.stress_field + np.random.normal(0, noise_level, self.stress_field.shape)
            
            # Adaptive Schwelle
            threshold = np.mean(noisy_field) + 2 * np.std(noisy_field)
            
            # Bereiche nahe Schwelle sind höchstes Risiko
            near_threshold = (noisy_field > threshold * 0.8) & (noisy_field <= threshold * 1.2)
            risk_accumulator[near_threshold] += 1
        
        # Normalisiere
        risk_map = risk_accumulator / 50
        
        # Glätte für regionale Muster
        risk_map = gaussian_filter(risk_map, sigma=2)
        
        return risk_map
    
    def calculate_criticality_metrics(self):
        """
        Berechne Metriken die Nähe zum kritischen Punkt anzeigen
        """
        metrics = {}
        
        # 1. Order Parameter (mittlerer Stress)
        metrics['order_parameter'] = np.mean(self.stress_field)
        
        # 2. Susceptibility (Varianz)
        metrics['susceptibility'] = np.var(self.stress_field)
        
        # 3. Korrelationslänge
        metrics['correlation_length'] = self._calculate_correlation_length()
        
        # 4. b-Wert (wenn genug Events)
        recent_mags = self.catalog.tail(100)['mag'].values
        if len(recent_mags) >= 50:
            metrics['b_value'] = self._calculate_b_value(recent_mags)
        else:
            metrics['b_value'] = 1.0
        
        # 5. Seismische Quiescence
        metrics['quiescence'] = self._calculate_quiescence()
        
        # 6. Criticality Index (kombiniert)
        metrics['criticality_index'] = (
            metrics['susceptibility'] * 
            metrics['correlation_length'] / 
            (metrics['b_value'] + 0.1)
        )
        
        return metrics
    
    def _calculate_correlation_length(self):
        """
        Berechne räumliche Korrelationslänge
        """
        # 2D Autokorrelation
        correlation = signal.correlate2d(
            self.stress_field, 
            self.stress_field, 
            mode='same'
        )
        
        # Normalisiere
        center_val = correlation[self.n_lat//2, self.n_lon//2]
        if center_val > 0:
            correlation /= center_val
        
        # Finde Abfall-Länge
        center = (self.n_lat//2, self.n_lon//2)
        
        for radius in range(1, min(self.n_lat, self.n_lon)//2):
            mask_sum = 0
            mask_count = 0
            
            for i in range(self.n_lat):
                for j in range(self.n_lon):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                    if abs(dist - radius) < 1:
                        mask_sum += correlation[i, j]
                        mask_count += 1
            
            if mask_count > 0:
                mean_correlation = mask_sum / mask_count
                if mean_correlation < 1/np.e:
                    return radius * self.grid_resolution  # In km
        
        return self.grid_resolution * min(self.n_lat, self.n_lon) / 4
    
    def _calculate_b_value(self, magnitudes):
        """
        Berechne Gutenberg-Richter b-Wert
        """
        if len(magnitudes) < 10:
            return 1.0
        
        mc = np.percentile(magnitudes, 10)  # Completeness magnitude
        complete = magnitudes[magnitudes >= mc]
        
        if len(complete) >= 5:
            b = np.log10(np.e) / (np.mean(complete) - mc + 0.1)
            return np.clip(b, 0.5, 2.0)
        
        return 1.0
    
    def _calculate_quiescence(self):
        """
        Berechne seismische Quiescence (Ruhe vor dem Sturm)
        """
        # Vergleiche aktuelle Rate mit historischem Durchschnitt
        if len(self.catalog) < 100:
            return 0.0
        
        # Letzte 30 Tage vs vorherige 90 Tage (mit timezone-aware timedelta)
        now = self.catalog['time'].max()
        
        recent = self.catalog[
            self.catalog['time'] > now - pd.Timedelta(days=30)
        ]
        
        historical = self.catalog[
            (self.catalog['time'] > now - pd.Timedelta(days=120)) &
            (self.catalog['time'] <= now - pd.Timedelta(days=30))
        ]
        
        if len(historical) > 0:
            recent_rate = len(recent) / 30
            historical_rate = len(historical) / 90
            
            quiescence = 1 - (recent_rate / (historical_rate + 0.1))
            return np.clip(quiescence, -1, 1)
        
        return 0.0
    
    def temporal_evolution(self, start_date, end_date, window_days=30, step_days=7):
        """
        Analysiere zeitliche Evolution des Systems
        """
        print("\nAnalysiere zeitliche Evolution...")
        
        # Konvertiere zu UTC-aware datetime
        current_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')
        
        # Verwende original_catalog für vollständige temporale Analyse
        if hasattr(self, 'original_catalog'):
            full_catalog = self.original_catalog
        else:
            full_catalog = self.catalog
        
        evolution_data = []
        
        with tqdm(total=(end_date-current_date).days//step_days) as pbar:
            while current_date < end_date:
                # Lade Daten für aktuelles Fenster
                window_start = current_date - pd.Timedelta(days=window_days)
                
                window_catalog = full_catalog[
                    (full_catalog['time'] >= window_start) &
                    (full_catalog['time'] <= current_date)
                ]
                
                if len(window_catalog) >= 10:
                    # Temporärer Katalog
                    self.catalog = window_catalog
                    
                    # Berechne Stress-Feld
                    self.calculate_stress_field(window_days)
                    
                    # Berechne Metriken
                    metrics = self.calculate_criticality_metrics()
                    metrics['date'] = current_date
                    
                    evolution_data.append(metrics)
                
                current_date += pd.Timedelta(days=step_days)
                pbar.update(1)
        
        # Restore original catalog
        self.catalog = full_catalog
        
        self.evolution_data = pd.DataFrame(evolution_data)
        return self.evolution_data
    
    def visualize_comprehensive_analysis(self):
        """
        Erstelle umfassende Visualisierung
        """
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Stress-Feld
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.imshow(
            self.stress_field.T, 
            origin='lower',
            extent=[
                self.region_bounds['lon_min'],
                self.region_bounds['lon_max'],
                self.region_bounds['lat_min'],
                self.region_bounds['lat_max']
            ],
            cmap='RdYlBu_r',
            aspect='auto'
        )
        ax1.set_title('Stress-Akkumulation')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(im1, ax=ax1, label='Normalisierter Stress')
        
        # 2. SR-verstärkte Risiko-Map
        ax2 = plt.subplot(3, 4, 2)
        im2 = ax2.imshow(
            self.risk_map.T,
            origin='lower',
            extent=[
                self.region_bounds['lon_min'],
                self.region_bounds['lon_max'],
                self.region_bounds['lat_min'],
                self.region_bounds['lat_max']
            ],
            cmap='Reds',
            aspect='auto'
        )
        ax2.set_title('SR-verstärkte Risikozone')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(im2, ax=ax2, label='Risiko-Level')
        
        # 3. Seismizität (letzte Events)
        ax3 = plt.subplot(3, 4, 3)
        recent = self.catalog.tail(1000)
        scatter = ax3.scatter(
            recent['longitude'],
            recent['latitude'],
            s=recent['mag']**2,
            c=recent['mag'],
            cmap='hot',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        ax3.set_xlim(self.region_bounds['lon_min'], self.region_bounds['lon_max'])
        ax3.set_ylim(self.region_bounds['lat_min'], self.region_bounds['lat_max'])
        ax3.set_title('Rezente Seismizität')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax3, label='Magnitude')
        
        # 4. Kombinierte Gefahrenkarte
        ax4 = plt.subplot(3, 4, 4)
        combined_hazard = self.stress_field * 0.5 + self.risk_map * 0.5
        im4 = ax4.imshow(
            combined_hazard.T,
            origin='lower',
            extent=[
                self.region_bounds['lon_min'],
                self.region_bounds['lon_max'],
                self.region_bounds['lat_min'],
                self.region_bounds['lat_max']
            ],
            cmap='magma',
            aspect='auto'
        )
        
        # Überlagere große Events
        large_events = self.catalog[self.catalog['mag'] >= 5.0]
        if len(large_events) > 0:
            ax4.scatter(
                large_events['longitude'],
                large_events['latitude'],
                s=100,
                c='cyan',
                marker='*',
                edgecolors='white',
                linewidth=1,
                label=f'M5+ Events (n={len(large_events)})'
            )
            ax4.legend()
        
        ax4.set_title('Kombinierte Gefahrenbewertung')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        plt.colorbar(im4, ax=ax4, label='Gefährdung')
        
        # 5. Zeitliche Evolution (wenn vorhanden)
        if hasattr(self, 'evolution_data') and len(self.evolution_data) > 0:
            ax5 = plt.subplot(3, 4, 5)
            ax5.plot(
                self.evolution_data['date'],
                self.evolution_data['criticality_index'],
                'b-',
                linewidth=2
            )
            ax5.axhline(
                y=self.evolution_data['criticality_index'].mean() + 
                  2*self.evolution_data['criticality_index'].std(),
                color='r',
                linestyle='--',
                label='Kritische Schwelle'
            )
            ax5.set_xlabel('Zeit')
            ax5.set_ylabel('Criticality Index')
            ax5.set_title('System nähert sich kritischem Punkt')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Rotation für Datum-Labels
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. SR Resonanzkurve
        ax6 = plt.subplot(3, 4, 6)
        if hasattr(self, 'sr_results'):
            noise_levels = [r['noise'] for r in self.sr_results]
            snr_values = [r['mean_snr'] for r in self.sr_results]
            snr_std = [r['std_snr'] for r in self.sr_results]
            
            ax6.semilogx(noise_levels, snr_values, 'g-', linewidth=2)
            ax6.fill_between(
                noise_levels,
                np.array(snr_values) - np.array(snr_std),
                np.array(snr_values) + np.array(snr_std),
                alpha=0.3,
                color='green'
            )
            ax6.axvline(
                self.optimal_noise,
                color='r',
                linestyle='--',
                label=f'Optimal: {self.optimal_noise:.4f}'
            )
            ax6.axvline(
                0.117,
                color='blue',
                linestyle=':',
                label='Theoretisches σc'
            )
            ax6.set_xlabel('Noise Level')
            ax6.set_ylabel('SNR')
            ax6.set_title('SR Resonanzkurve')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Magnitude-Zeit Diagramm
        ax7 = plt.subplot(3, 4, 7)
        ax7.scatter(
            self.catalog['time'],
            self.catalog['mag'],
            s=self.catalog['mag']**2,
            alpha=0.5,
            c=self.catalog['mag'],
            cmap='viridis'
        )
        ax7.set_xlabel('Zeit')
        ax7.set_ylabel('Magnitude')
        ax7.set_title('Seismische Aktivität')
        ax7.grid(True, alpha=0.3)
        
        # 8. b-Wert Analyse
        ax8 = plt.subplot(3, 4, 8)
        magnitudes = self.catalog['mag'].values
        mag_bins = np.arange(
            np.floor(magnitudes.min()),
            np.ceil(magnitudes.max()) + 0.1,
            0.1
        )
        hist, bins = np.histogram(magnitudes, bins=mag_bins)
        
        # Kumulative Verteilung
        cumulative = np.array([np.sum(hist[i:]) for i in range(len(hist))])
        
        # Log-Plot für Gutenberg-Richter
        valid = cumulative > 0
        ax8.semilogy(
            bins[:-1][valid],
            cumulative[valid],
            'bo',
            markersize=4,
            label='Beobachtet'
        )
        
        # Fit Gutenberg-Richter
        if np.sum(valid) > 3:
            mc = np.percentile(magnitudes, 10)
            fit_mask = (bins[:-1] >= mc) & valid
            if np.sum(fit_mask) > 2:
                fit = np.polyfit(
                    bins[:-1][fit_mask],
                    np.log10(cumulative[fit_mask]),
                    1
                )
                fit_line = 10**(fit[0] * bins[:-1] + fit[1])
                ax8.semilogy(
                    bins[:-1],
                    fit_line,
                    'r-',
                    label=f'G-R Fit (b={-fit[0]:.2f})'
                )
        
        ax8.set_xlabel('Magnitude')
        ax8.set_ylabel('Kumulative Anzahl')
        ax8.set_title('Gutenberg-Richter Verteilung')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Kritikalitäts-Metriken
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        metrics = self.calculate_criticality_metrics()
        
        text = f"""Kritikalitäts-Analyse
{'='*30}
Order Parameter: {metrics['order_parameter']:.3f}
Susceptibility: {metrics['susceptibility']:.3f}
Korrelation [km]: {metrics['correlation_length']:.1f}
b-Wert: {metrics['b_value']:.2f}
Quiescence: {metrics['quiescence']:.2f}
{'='*30}
Criticality Index: {metrics['criticality_index']:.2f}

Status: """
        
        if metrics['criticality_index'] > 10:
            text += "🚨 KRITISCH!"
            color = 'red'
        elif metrics['criticality_index'] > 5:
            text += "⚠️ ERHÖHTES RISIKO"
            color = 'orange'
        else:
            text += "✅ Normal"
            color = 'green'
        
        ax9.text(
            0.1, 0.9, text,
            transform=ax9.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace'
        )
        
        # Farbiger Hintergrund basierend auf Status
        ax9.add_patch(
            plt.Rectangle(
                (0, 0), 1, 1,
                transform=ax9.transAxes,
                alpha=0.2,
                facecolor=color
            )
        )
        
        # 10. Hochrisiko-Zonen Identifikation
        ax10 = plt.subplot(3, 4, 10)
        
        # Finde Top 5 Risiko-Hotspots
        risk_threshold = np.percentile(self.risk_map, 95)
        high_risk = self.risk_map > risk_threshold
        
        # Zeige nur Hochrisiko-Bereiche
        masked_risk = np.ma.masked_where(~high_risk, self.risk_map)
        
        im10 = ax10.imshow(
            masked_risk.T,
            origin='lower',
            extent=[
                self.region_bounds['lon_min'],
                self.region_bounds['lon_max'],
                self.region_bounds['lat_min'],
                self.region_bounds['lat_max']
            ],
            cmap='Reds',
            aspect='auto',
            vmin=risk_threshold,
            vmax=self.risk_map.max()
        )
        
        ax10.set_title('Hochrisiko-Zonen (Top 5%)')
        ax10.set_xlabel('Longitude')
        ax10.set_ylabel('Latitude')
        plt.colorbar(im10, ax=ax10, label='Risiko')
        
        # 11. 3D Stress-Landschaft
        ax11 = fig.add_subplot(3, 4, 11, projection='3d')
        
        # Downsample für Performance
        step = max(1, min(self.n_lat, self.n_lon) // 30)
        lat_indices = np.arange(0, self.n_lat, step)
        lon_indices = np.arange(0, self.n_lon, step)
        
        X, Y = np.meshgrid(
            self.lon_grid[lon_indices],
            self.lat_grid[lat_indices]
        )
        Z = self.stress_field[np.ix_(lat_indices, lon_indices)]
        
        surf = ax11.plot_surface(
            X, Y, Z,
            cmap='viridis',
            alpha=0.8,
            antialiased=True
        )
        
        ax11.set_xlabel('Longitude')
        ax11.set_ylabel('Latitude')
        ax11.set_zlabel('Stress')
        ax11.set_title('3D Stress-Landschaft')
        ax11.view_init(elev=20, azim=45)
        
        # 12. Vorhersage-Konfidenz
        ax12 = plt.subplot(3, 4, 12)
        
        # Berechne Vorhersage-Konfidenz basierend auf mehreren Faktoren
        confidence_factors = {
            'Datenqualität': min(len(self.catalog) / 1000, 1.0),
            'Zeitliche Abdeckung': min((self.catalog['time'].max() - self.catalog['time'].min()).days / 365, 1.0),
            'Räumliche Auflösung': min(self.n_lat * self.n_lon / 100, 1.0),
            'SR Signal': min(self.sr_results[np.argmax([r['mean_snr'] for r in self.sr_results])]['mean_snr'] / 5, 1.0),
            'Kritikalität': min(metrics['criticality_index'] / 10, 1.0)
        }
        
        labels = list(confidence_factors.keys())
        values = list(confidence_factors.values())
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_plot = values + [values[0]]  # Schließe den Kreis
        angles += angles[:1]
        
        ax12 = plt.subplot(3, 4, 12, projection='polar')
        ax12.plot(angles, values_plot, 'o-', linewidth=2, color='blue')
        ax12.fill(angles, values_plot, alpha=0.25, color='blue')
        ax12.set_xticks(angles[:-1])
        ax12.set_xticklabels(labels, size=8)
        ax12.set_ylim(0, 1)
        ax12.set_title('Vorhersage-Konfidenz', y=1.08)
        ax12.grid(True)
        
        # Gesamt-Titel
        plt.suptitle(
            f'Makro-Skala SR Erdbebenanalyse | Region: {self.region_bounds} | '
            f'Optimal Noise: {self.optimal_noise:.4f} | '
            f'Criticality: {metrics["criticality_index"]:.2f}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return fig
    
    def save_analysis(self, filename_prefix='earthquake_sr_analysis'):
        """
        Speichere Analyse-Ergebnisse
        """
        # Speichere Plots
        fig = self.visualize_comprehensive_analysis()
        fig.savefig(f'{filename_prefix}_visualization.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Speichere Daten
        save_data = {
            'region_bounds': self.region_bounds,
            'grid_resolution': self.grid_resolution,
            'optimal_noise': self.optimal_noise,
            'stress_field': self.stress_field.tolist(),
            'risk_map': self.risk_map.tolist(),
            'criticality_metrics': self.calculate_criticality_metrics(),
            'catalog_summary': {
                'n_events': len(self.catalog),
                'date_range': [
                    self.catalog['time'].min().isoformat(),
                    self.catalog['time'].max().isoformat()
                ],
                'magnitude_range': [
                    float(self.catalog['mag'].min()),
                    float(self.catalog['mag'].max())
                ]
            }
        }
        
        # JSON für Metriken
        with open(f'{filename_prefix}_metrics.json', 'w') as f:
            json_data = {k: v for k, v in save_data.items() 
                        if not isinstance(v, (np.ndarray, list)) or k in ['criticality_metrics', 'catalog_summary']}
            json.dump(json_data, f, indent=2, default=str)
        
        # Pickle für Arrays
        with open(f'{filename_prefix}_data.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nAnalyse gespeichert als:")
        print(f"  - {filename_prefix}_visualization.png")
        print(f"  - {filename_prefix}_metrics.json")
        print(f"  - {filename_prefix}_data.pkl")


def run_complete_analysis(region='california'):
    """
    Führe komplette Analyse für eine Region durch
    """
    print("="*70)
    print("MAKRO-SKALA SR SYSTEM FÜR ERDBEBENVORHERSAGE")
    print("="*70)
    
    # Definiere Regionen
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
        },
        'turkey': {
            'lat_min': 36.0,
            'lat_max': 42.0,
            'lon_min': 26.0,
            'lon_max': 45.0
        }
    }
    
    if region not in regions:
        print(f"Unbekannte Region: {region}")
        region = 'california'
    
    print(f"\nAnalysiere Region: {region.upper()}")
    print(f"Grenzen: {regions[region]}")
    
    # Initialisiere System
    system = MacroScaleSRSystem(
        region_bounds=regions[region],
        grid_resolution_km=50
    )
    
    # Lade Erdbebenkatalog
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 Jahre
    
    catalog = system.load_earthquake_catalog(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=2.5
    )
    
    # Berechne Stress-Feld
    system.calculate_stress_field(time_window_days=90)
    
    # Wende SR an
    system.sr_results = system.apply_stochastic_resonance()
    
    # Zeitliche Evolution (optional, dauert länger)
    print("\nMöchten Sie die zeitliche Evolution analysieren? (dauert ~5-10 min)")
    print("Geben Sie 'ja' ein für zeitliche Analyse, oder Enter zum Überspringen:")
    
    user_input = input().strip().lower()
    
    if user_input == 'ja':
        evolution = system.temporal_evolution(
            start_date=start_date,
            end_date=end_date,
            window_days=90,
            step_days=30
        )
        
        # Zeige Evolution
        if len(evolution) > 0:
            print("\nZeitliche Evolution des Criticality Index:")
            print(evolution[['date', 'criticality_index']].tail(10))
    
    # Berechne finale Metriken
    metrics = system.calculate_criticality_metrics()
    
    # Zeige Ergebnisse
    print("\n" + "="*70)
    print("ANALYSE-ERGEBNISSE")
    print("="*70)
    
    print(f"\nOptimales Noise-Level: {system.optimal_noise:.6f}")
    print(f"Verhältnis zu σc: {system.optimal_noise/0.117:.2f}x")
    
    print(f"\nKritikalitäts-Metriken:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Risikobewertung
    print(f"\n{'='*40}")
    if metrics['criticality_index'] > 10:
        print("🚨 WARNUNG: System zeigt KRITISCHE Anzeichen!")
        print("   Erhöhtes Risiko für größeres Event")
    elif metrics['criticality_index'] > 5:
        print("⚠️  ERHÖHTE AUFMERKSAMKEIT empfohlen")
        print("   System nähert sich kritischem Punkt")
    else:
        print("✅ System im NORMALEN Bereich")
        print("   Kein unmittelbares erhöhtes Risiko")
    print('='*40)
    
    # Visualisierung
    print("\nErstelle Visualisierungen...")
    system.visualize_comprehensive_analysis()
    
    # Speichern
    system.save_analysis(f'{region}_sr_analysis')
    
    print("\n" + "="*70)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*70)
    
    return system


if __name__ == "__main__":
    # Führe Analyse für California durch
    system = run_complete_analysis('california')
    
    # Optional: Weitere Regionen
    # system_japan = run_complete_analysis('japan')
    # system_chile = run_complete_analysis('chile')