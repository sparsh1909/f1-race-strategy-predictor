import fastf1
import pandas as pd
import os

# Create cache folder
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Load 2023 Abu Dhabi Grand Prix
session = fastf1.get_session(2023, 'Abu Dhabi', 'R')
session.load()

# Get lap data
laps = session.laps
print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'PitOutTime', 'PitInTime']].head(20))
print("\nShape:", laps.shape)
print("\nDrivers:", laps['Driver'].unique())