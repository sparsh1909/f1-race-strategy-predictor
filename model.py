import sys
print("Python:", sys.version)
print("Script started")

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2023, 'Abu Dhabi', 'R')
session.load()

laps = session.laps.copy()
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
laps = laps[laps['LapTimeSeconds'] > 85]
laps = laps[laps['LapTimeSeconds'] < 120]
laps = laps.dropna(subset=['LapTimeSeconds', 'Compound', 'TyreLife'])
laps = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

laps = laps.sort_values(['Driver', 'LapNumber'])
laps['PittedNextLap'] = laps.groupby('Driver')['PitInTime'].shift(-1).notna().astype(int)

compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
laps['CompoundEncoded'] = laps['Compound'].map(compound_map)

features = ['LapNumber', 'TyreLife', 'CompoundEncoded', 'LapTimeSeconds']
X = laps[features].dropna()
y = laps.loc[X.index, 'PittedNextLap']

print("Dataset shape:", X.shape)
print("Pit stops in data:", y.sum(), "out of", len(y), "laps")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='#E8002D')
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')
ax.set_title('What Predicts a Pit Stop?', color='white')
ax.tick_params(colors='white')
ax.set_xlabel('Feature Importance', color='white')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
print("\nPlot saved!")
plt.show()