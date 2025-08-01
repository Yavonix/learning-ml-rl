import numpy as np
import matplotlib.pyplot as plt

# Given data
times = np.array([4, 11, 13, 24, 30])  # hours
concs = np.array([4.5, 1.7, 1.45, 0.33, 0.16])  # mg/L
dose = 200  # mg

# Perform log-linear regression
ln_concs = np.log(concs)
slope, intercept = np.polyfit(times, ln_concs, 1)

# Extrapolated C0 and volume of distribution
C0 = np.exp(intercept)

print(f"Extrapolated C0: {C0:.2f} mg/L")
print(f"Dose: {dose:.2f} ")

Vd = dose / C0  # in liters

# Plotting
plt.plot(times, concs, marker='o')
t_fit = np.linspace(0, 30, 100)
conc_fit = np.exp(intercept + slope * t_fit)
plt.plot(t_fit, conc_fit)
plt.xlabel('Time (h)')
plt.ylabel('Drug concentration (mg/L)')
plt.title('Concentration vs Time with Exponential Fit')
plt.show()

print(f"Volume of distribution: {Vd:.2f} L")
