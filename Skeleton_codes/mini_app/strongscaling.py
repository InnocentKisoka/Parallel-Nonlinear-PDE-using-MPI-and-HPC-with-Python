import matplotlib.pyplot as plt
import numpy as np

# Data
ncpu = [1, 2, 4, 8, 16]
grid_sizes = ["64×64", "128×128", "256×256", "512×512", "1024×1024"]
times = [
    [0.0398729, 0.0229523, 0.0147691, 0.012604, 0.0153703],  # 64×64
    [0.259799, 0.130442, 0.0686735, 0.0404601, 0.0315571],  # 128×128
    [1.71422, 0.878382, 0.454705, 0.230752, 0.134358],      # 256×256
    [13.1958, 6.31247, 3.25356, 1.67867, 0.84599],          # 512×512
    [153.788, 83.1634, 54.1292, 42.5365, 21.9804],          # 1024×1024
]

# Plot
plt.figure(figsize=(10, 6))
for i, grid in enumerate(grid_sizes):
    plt.plot(ncpu, times[i], marker='o', label=grid)

# Logarithmic scale for better visualization
plt.yscale("log")

# Labels and legend
plt.xlabel("Number of CPUs (NCPU)")
plt.ylabel("Execution Time (seconds, log scale)")
plt.title("Strong Scaling: Time vs Number of CPUs")
plt.legend(title="Grid Size")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("Strong Scaling Plot")
plt.show()
