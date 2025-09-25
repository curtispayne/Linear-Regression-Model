import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Loading Session Data
session = fastf1.get_session(2025,"Bahrain","FP2")
session.load(laps=True,telemetry=True)
lap_data = session.laps.pick_driver("NOR")

# Ensuring Correct Run Data Used
lap_data = lap_data[lap_data["Stint"]==3.0]

# Adding Delta Column With Reference To Stint Minimum Laptime
lap_data["Delta"] = lap_data.LapTime.dt.total_seconds() - lap_data.LapTime.dt.total_seconds().min()

# Excluding Invalid Laps
lap_data = lap_data[lap_data.LapTime.notna()]

# Removing Anomalies Using Median
med = np.median(lap_data.Delta)
lap_data = lap_data[lap_data.Delta<=2*med]

# Converting Lap Time Column to Seconds
lap_data.LapTime = lap_data.LapTime.dt.total_seconds()

# Performing Linear Regression For Each Stint
m,c = np.polyfit(lap_data.TyreLife,lap_data.LapTime,1)

# Calculating Predicted v Actual Difference
pred = m*lap_data.LapNumber+c
lap_data["Difference"] = lap_data.LapTime - pred
std = np.std(pred)

# Print Deg and Std Values
print("Deg: " + str(round(m,3)) + " | Std: " + str(round(std,3)) +" | Initial Lap Time: " + str(round(c,3)) + " | Tyre: " + lap_data.iloc[0]["Compound"])

# Plotting Each Stint For a Visual Guide
plt.plot(lap_data.TyreLife,lap_data.LapTime,marker="o",color="r")
plt.xlabel("Lap Number")
plt.ylabel("Delta to Fastest Stint Lap [s]")
plt.title("Tyre: " + lap_data.iloc[0]["Compound"])
plt.show()