import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('cache', exist_ok=True)
os.makedirs('plots', exist_ok=True)
fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2023, 'Abu Dhabi', 'R')
session.load()

laps = session.laps.copy()

# Convert LapTime to seconds
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Clean data - remove outliers and safety car laps
laps = laps[laps['LapTimeSeconds'] > 0]
laps = laps[laps['LapTimeSeconds'] < 120]  # Abu Dhabi lap ~87s, remove obvious outliers
laps = laps.dropna(subset=['LapTimeSeconds', 'Compound', 'TyreLife'])

# Keep only racing compounds
laps = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

print("Clean laps shape:", laps.shape)
print("\nCompound distribution:\n", laps['Compound'].value_counts())
print("\nAvg lap time by compound:")
print(laps.groupby('Compound')['LapTimeSeconds'].mean().round(2))

# Plot lap time degradation by compound
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}

for compound in ['SOFT', 'MEDIUM', 'HARD']:
    data = laps[laps['Compound'] == compound]
    avg_by_age = data.groupby('TyreLife')['LapTimeSeconds'].median()
    ax.plot(avg_by_age.index, avg_by_age.values, 
            label=compound, color=colors[compound], linewidth=2)

ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')
ax.set_xlabel('Tyre Age (Laps)', color='white')
ax.set_ylabel('Lap Time (seconds)', color='white')
ax.set_title('Tyre Degradation by Compound - 2023 Abu Dhabi GP', color='white')
ax.tick_params(colors='white')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/tyre_degradation.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to plots/tyre_degradation.png")
plt.show()