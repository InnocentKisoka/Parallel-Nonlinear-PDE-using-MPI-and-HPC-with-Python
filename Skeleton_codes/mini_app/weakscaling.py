import matplotlib.pyplot as plt
import numpy as np

# Data from the table
NCPU = np.array([1, 2, 4, 8, 16])
times_64 = np.array([0.0392514, 0.0498204, 0.0676643, 0.0885274, 0.134214])
times_128 = np.array([0.262674, 0.331769, 0.452449, 0.613271, 0.846193])
times_256 = np.array([1.71919, 2.32069, 3.25216, 10.9703, 21.7381])

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(NCPU, times_64, label="Resolution 64x64", marker='o', color='b')
plt.plot(NCPU, times_128, label="Resolution 128x128", marker='o', color='orange')
plt.plot(NCPU, times_256, label="Resolution 256x256", marker='o', color='r')

# Set log scale for y-axis
plt.yscale('log')

# Labels and Title
plt.xlabel('Number of Processes (NCPU)')
plt.ylabel('Time to Solution (seconds, log scale)')
plt.title('Weak Scaling Performance for Different Resolutions')

# Show grid and legen
plt.grid(True, which="both", ls="--")
plt.legend()

# Show plot
plt.savefig("Weak Scaling plot")
plt.show()
