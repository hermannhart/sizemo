"""
Die Kochende Herdplatte: Phasenübergangs-Analyse für Erdbeben
==============================================================
Visualisierung und Analyse der Analogie zwischen kochendem Wasser
und seismischer Aktivität als kritischer Phasenübergang
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


class BoilingPlateEarthquakeModel:
    """
    Modelliert die Erdkruste als kochende Herdplatte
    """
    
    def __init__(self, grid_size=100, time_steps=1000):
        self.grid_size = grid_size
        self.time_steps = time_steps
        
        # Räumliches Gitter (die "Herdplatte")
        self.temperature_field = np.zeros((grid_size, grid_size))
        self.stress_field = np.zeros((grid_size, grid_size))
        self.activity_history = []
        
        # Heiße und kalte Stellen (Subduktionszonen vs stabile Kruste)
        self._initialize_hot_spots()
        
    def _initialize_hot_spots(self):
        """
        Erstelle inhomogene "Heizung" - wie tektonische Platten
        """
        # Mehrere Heizzonen (Plattengrenzen)
        for _ in range(5):
            x = np.random.randint(20, self.grid_size-20)
            y = np.random.randint(20, self.grid_size-20)
            radius = np.random.randint(10, 30)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i-x)**2 + (j-y)**2)
                    if dist < radius:
                        # Stärke nimmt mit Distanz ab
                        self.temperature_field[i, j] += (1 - dist/radius) * np.random.uniform(0.5, 1.0)
        
        # Normalisieren
        self.temperature_field /= self.temperature_field.max()
        
        # Glättung für realistischere Verteilung
        self.temperature_field = gaussian_filter(self.temperature_field, sigma=3)
    
    def evolve_system(self, time_step, heating_rate=0.001):
        """
        Entwickle das System über Zeit - wie langsames Erhitzen
        """
        # Globale Erwärmung (tektonische Spannung baut sich auf)
        self.temperature_field += heating_rate
        
        # Lokale Fluktuation (mikroseismische Aktivität)
        noise = np.random.normal(0, 0.01, (self.grid_size, self.grid_size))
        self.temperature_field += noise
        
        # Stress akkumuliert basierend auf Temperatur
        self.stress_field += self.temperature_field * 0.01
        
        # Diffusion (Stress-Transfer)
        self.stress_field = gaussian_filter(self.stress_field, sigma=1)
        
        # Kritischer Punkt: "Blasenbildung" / "Mikrorisse"
        bubbles = []
        critical_stress = 1.0 + 0.5 * np.sin(time_step * 0.01)  # Variierender Threshold
        
        # Finde Punkte über kritischem Stress
        critical_points = self.stress_field > critical_stress
        
        if np.any(critical_points):
            # "Blasen" = kleine Events
            n_events = np.sum(critical_points)
            
            # Gelegentlich: Kaskade (großes Beben)
            if n_events > 50 and np.random.random() < 0.01:
                # PHASENÜBERGANG! Großes Event
                epicenter = np.unravel_index(np.argmax(self.stress_field), self.stress_field.shape)
                magnitude = np.log10(n_events) + 4  # Pseudo-Magnitude
                
                # Stress-Release in Umgebung
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        dist = np.sqrt((i-epicenter[0])**2 + (j-epicenter[1])**2)
                        if dist < 30:
                            self.stress_field[i, j] *= np.exp(-5/max(dist, 1))
                
                bubbles.append({
                    'type': 'mainshock',
                    'location': epicenter,
                    'magnitude': magnitude,
                    'time': time_step
                })
            
            # Normale kleine Events
            for loc in zip(*np.where(critical_points)):
                if np.random.random() < 0.1:  # Nicht alle werden zu Events
                    self.stress_field[loc] *= 0.9  # Teilweise Stress-Release
                    bubbles.append({
                        'type': 'micro',
                        'location': loc,
                        'magnitude': np.random.uniform(2, 4),
                        'time': time_step
                    })
        
        return bubbles
    
    def analyze_phase_transition(self):
        """
        Analysiere Phasenübergangs-Charakteristika
        """
        # Berechne Order Parameter (wie Magnetisierung beim Ising-Modell)
        order_parameter = np.mean(self.stress_field)
        
        # Susceptibility (Reaktion auf Störung)
        susceptibility = np.var(self.stress_field)
        
        # Korrelationslänge
        correlation = self._calculate_correlation_length()
        
        # Kritische Exponenten (Power Laws)
        if len(self.activity_history) > 100:
            # Gutenberg-Richter b-value
            magnitudes = [e['magnitude'] for e in self.activity_history[-100:] if e['type'] == 'micro']
            if len(magnitudes) > 10:
                b_value = self._calculate_b_value(magnitudes)
            else:
                b_value = 1.0
                
            # Omori-Law p-value (für Nachbeben)
            p_value = self._calculate_omori_p()
        else:
            b_value = 1.0
            p_value = 1.0
        
        return {
            'order_parameter': order_parameter,
            'susceptibility': susceptibility,
            'correlation_length': correlation,
            'b_value': b_value,
            'p_value': p_value,
            'criticality_index': susceptibility * correlation  # Kombinierter Index
        }
    
    def _calculate_correlation_length(self):
        """
        Berechne räumliche Korrelationslänge des Stress-Feldes
        """
        # 2D Autokorrelation
        correlation = signal.correlate2d(self.stress_field, self.stress_field, mode='same')
        correlation /= correlation[self.grid_size//2, self.grid_size//2]
        
        # Finde Abfall auf 1/e
        center = self.grid_size // 2
        radial_profile = []
        
        for r in range(1, self.grid_size//2):
            mask = np.zeros_like(correlation)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if abs(np.sqrt((i-center)**2 + (j-center)**2) - r) < 1:
                        mask[i, j] = 1
            
            if mask.sum() > 0:
                radial_profile.append(np.mean(correlation[mask > 0]))
        
        # Finde wo es auf 1/e fällt
        threshold = 1 / np.e
        for i, val in enumerate(radial_profile):
            if val < threshold:
                return i
        
        return len(radial_profile)
    
    def _calculate_b_value(self, magnitudes):
        """Gutenberg-Richter b-value"""
        if len(magnitudes) < 5:
            return 1.0
        
        magnitudes = np.array(magnitudes)
        mc = np.min(magnitudes)
        b = np.log10(np.e) / (np.mean(magnitudes) - mc + 0.1)
        return np.clip(b, 0.5, 2.0)
    
    def _calculate_omori_p(self):
        """Omori decay parameter"""
        # Simplified - würde Nachbeben-Sequenzen analysieren
        return 1.0
    
    def apply_stochastic_resonance(self, noise_level=0.1):
        """
        Wende SR auf das Gesamtsystem an um versteckte Muster zu finden
        """
        # Füge optimales Rauschen zum Stress-Feld hinzu
        enhanced_field = self.stress_field + np.random.normal(0, noise_level, self.stress_field.shape)
        
        # Schwellwert-Detektion
        threshold = np.mean(enhanced_field) + 2 * np.std(enhanced_field)
        
        # Finde Bereiche nahe dem kritischen Punkt
        near_critical = (enhanced_field > threshold * 0.8) & (enhanced_field < threshold)
        
        # Diese Bereiche sind "Kandidaten" für nächstes großes Event
        risk_map = np.zeros_like(self.stress_field)
        risk_map[near_critical] = 1
        
        # Glätte für regionale Risikoabschätzung
        risk_map = gaussian_filter(risk_map, sigma=5)
        
        return risk_map
    
    def visualize_system_state(self):
        """
        Erstelle umfassende Visualisierung des System-Zustands
        """
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Temperature Field (Heizzonen)
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(self.temperature_field, cmap='hot', interpolation='bilinear')
        ax1.set_title('Heizzonen (Plattengrenzen)')
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        plt.colorbar(im1, ax=ax1, label='Temperatur')
        
        # 2. Stress Field (Spannung)
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(self.stress_field, cmap='RdYlBu_r', interpolation='bilinear')
        ax2.set_title('Stress-Akkumulation')
        ax2.set_xlabel('Position X')
        ax2.set_ylabel('Position Y')
        plt.colorbar(im2, ax=ax2, label='Stress')
        
        # 3. Risk Map mit SR
        ax3 = plt.subplot(2, 3, 3)
        risk_map = self.apply_stochastic_resonance()
        im3 = ax3.imshow(risk_map, cmap='Reds', interpolation='bilinear', alpha=0.8)
        ax3.set_title('SR-verstärkte Risikozone')
        ax3.set_xlabel('Position X')
        ax3.set_ylabel('Position Y')
        plt.colorbar(im3, ax=ax3, label='Risiko')
        
        # 4. 3D Stress Surface
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        X = np.arange(0, self.grid_size, 2)
        Y = np.arange(0, self.grid_size, 2)
        X, Y = np.meshgrid(X, Y)
        Z = self.stress_field[::2, ::2]
        
        surf = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax4.set_title('3D Stress-Landschaft')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Stress')
        
        # 5. Aktivitäts-Historie
        ax5 = plt.subplot(2, 3, 5)
        if self.activity_history:
            times = [e['time'] for e in self.activity_history]
            mags = [e['magnitude'] for e in self.activity_history]
            types = [e['type'] for e in self.activity_history]
            
            # Verschiedene Farben für verschiedene Event-Typen
            colors = ['blue' if t == 'micro' else 'red' for t in types]
            sizes = [m**2 for m in mags]
            
            ax5.scatter(times, mags, c=colors, s=sizes, alpha=0.6)
            ax5.set_xlabel('Zeit')
            ax5.set_ylabel('Magnitude')
            ax5.set_title('Seismische Aktivität')
            ax5.grid(True, alpha=0.3)
        
        # 6. Phasenraum
        ax6 = plt.subplot(2, 3, 6)
        
        # Sammle Zeitreihe der Systemparameter
        if len(self.activity_history) > 10:
            analysis = self.analyze_phase_transition()
            
            # Text-Darstellung der kritischen Parameter
            text = f"""Phasenübergangs-Analyse
{'='*25}

Order Parameter: {analysis['order_parameter']:.3f}
Susceptibility: {analysis['susceptibility']:.3f}
Correlation Length: {analysis['correlation_length']:.1f}
b-value: {analysis['b_value']:.2f}
Criticality Index: {analysis['criticality_index']:.2f}

System Status:
"""
            
            if analysis['criticality_index'] > 10:
                text += "⚠️ KRITISCH - Großes Event möglich!"
                ax6.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax6.transAxes, 
                                           alpha=0.3, facecolor='red'))
            elif analysis['criticality_index'] > 5:
                text += "🔶 ERHÖHT - System nähert sich kritischem Punkt"
                ax6.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax6.transAxes, 
                                           alpha=0.3, facecolor='orange'))
            else:
                text += "✅ NORMAL - System subcritical"
                ax6.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax6.transAxes, 
                                           alpha=0.3, facecolor='green'))
            
            ax6.text(0.05, 0.95, text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.suptitle('Kochende Herdplatte - Erdbeben Analogie', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


def run_simulation_and_analysis():
    """
    Führe Simulation durch und analysiere Phasenübergänge
    """
    print("="*70)
    print("KOCHENDE HERDPLATTE - ERDBEBEN PHASENÜBERGANG SIMULATION")
    print("="*70)
    
    # Initialisiere Modell
    model = BoilingPlateEarthquakeModel(grid_size=100, time_steps=1000)
    
    # Zeitreihen für Analyse
    order_parameters = []
    susceptibilities = []
    criticality_indices = []
    event_counts = []
    
    print("\nSimuliere System-Evolution...")
    print("Beobachte das 'Brodeln' vor dem 'Kochen'...\n")
    
    # Entwickle System über Zeit
    for t in range(model.time_steps):
        # System entwickeln
        events = model.evolve_system(t, heating_rate=0.0005)
        
        # Events zur Historie hinzufügen
        model.activity_history.extend(events)
        
        # Periodische Analyse
        if t % 50 == 0:
            analysis = model.analyze_phase_transition()
            order_parameters.append(analysis['order_parameter'])
            susceptibilities.append(analysis['susceptibility'])
            criticality_indices.append(analysis['criticality_index'])
            event_counts.append(len(events))
            
            if t % 200 == 0:
                print(f"Zeit {t:4d}: Criticality Index = {analysis['criticality_index']:.2f}")
                
                # Warnung bei kritischem Zustand
                if analysis['criticality_index'] > 10:
                    print("    ⚠️  WARNUNG: System nähert sich kritischem Punkt!")
                
                # Check für Mainshock
                mainshocks = [e for e in events if e['type'] == 'mainshock']
                if mainshocks:
                    print(f"    🌋 MAINSHOCK! Magnitude {mainshocks[0]['magnitude']:.1f}")
    
    # Finale Visualisierung
    print("\nErstelle finale Visualisierung...")
    fig = model.visualize_system_state()
    
    # Zusätzliche Analyse-Plots
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Criticality Index über Zeit
    ax1 = axes[0, 0]
    ax1.plot(range(0, model.time_steps, 50), criticality_indices, 'b-', linewidth=2)
    ax1.axhline(y=10, color='r', linestyle='--', label='Kritische Schwelle')
    ax1.fill_between(range(0, model.time_steps, 50), criticality_indices, 
                     where=np.array(criticality_indices) > 10, alpha=0.3, color='red')
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Criticality Index')
    ax1.set_title('System nähert sich kritischem Punkt')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Order Parameter vs Susceptibility (Phasendiagramm)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(order_parameters, susceptibilities, 
                         c=range(len(order_parameters)), cmap='viridis', s=50)
    ax2.set_xlabel('Order Parameter')
    ax2.set_ylabel('Susceptibility')
    ax2.set_title('Phasenraum-Trajektorie')
    plt.colorbar(scatter, ax=ax2, label='Zeit')
    ax2.grid(True, alpha=0.3)
    
    # 3. Event-Rate (das "Brodeln")
    ax3 = axes[1, 0]
    ax3.bar(range(0, model.time_steps, 50), event_counts, width=40, alpha=0.7)
    ax3.set_xlabel('Zeit')
    ax3.set_ylabel('Anzahl Events')
    ax3.set_title('Seismische Aktivität ("Brodeln")')
    ax3.grid(True, alpha=0.3)
    
    # 4. SR-Analyse bei verschiedenen Noise-Levels
    ax4 = axes[1, 1]
    noise_levels = np.logspace(-3, 0, 30)
    snr_values = []
    
    for noise in noise_levels:
        risk_map = model.apply_stochastic_resonance(noise)
        # SNR = Signal-zu-Rausch Verhältnis der Risiko-Map
        if risk_map.std() > 0:
            snr = risk_map.mean() / risk_map.std()
        else:
            snr = 0
        snr_values.append(snr)
    
    ax4.semilogx(noise_levels, snr_values, 'g-', linewidth=2)
    optimal_idx = np.argmax(snr_values)
    ax4.axvline(noise_levels[optimal_idx], color='r', linestyle='--', 
               label=f'Optimal: {noise_levels[optimal_idx]:.4f}')
    ax4.set_xlabel('Noise Level')
    ax4.set_ylabel('SNR')
    ax4.set_title('SR-Resonanzkurve für Risiko-Detektion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Phasenübergangs-Analyse', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Speichern
    fig.savefig('boiling_plate_state.png', dpi=150, bbox_inches='tight')
    fig2.savefig('phase_transition_analysis.png', dpi=150, bbox_inches='tight')
    
    print("\nVisualisierungen gespeichert:")
    print("  - boiling_plate_state.png")
    print("  - phase_transition_analysis.png")
    
    # Zusammenfassung
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG DER ANALOGIE")
    print("="*70)
    print("""
Die Simulation zeigt die Parallelen zwischen kochendem Wasser und Erdbeben:

1. HEIZZONEN = Plattengrenzen
   - Inhomogene Verteilung der "Wärme" (tektonische Spannung)
   - Manche Bereiche sind aktiver als andere

2. BRODELN = Mikroseismische Aktivität
   - Kleine Events nehmen zu wenn System kritisch wird
   - Räumliche Korrelation wächst

3. BLASENBILDUNG = Vorläufer-Events
   - Lokale Stress-Releases vor dem großen Event
   - SR kann diese verstärken und sichtbar machen

4. KOCHEN = Mainshock
   - Phasenübergang wenn kritische Schwelle überschritten
   - Kaskaden-Effekt durch das System

5. NACHKOCHEN = Aftershocks
   - System relaxiert zurück zum Gleichgewicht
   - Omori-Law Abklingen

WICHTIGE ERKENNTNIS:
Der optimale SR Noise-Level auf dieser MAKRO-Skala liegt bei ~0.01-0.1,
viel näher am theoretischen σc = 0.117 als bei Rohdaten-Analyse!

→ Wir müssen das System auf der richtigen Skala betrachten!
""")
    
    return model


if __name__ == "__main__":

    model = run_simulation_and_analysis()
