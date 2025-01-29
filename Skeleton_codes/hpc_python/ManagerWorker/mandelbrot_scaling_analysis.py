import matplotlib.pyplot as plt
import numpy as np

# Data from the table
ranks = [2, 4, 8, 16]
tasks_50_time = [24.5874, 8.6955, 4.4085, 2.2157]
tasks_100_time = [24.8438, 8.4655, 3.9515, 1.8918]
tasks_50_speedup = [1, 2.8277, 5.5773, 11.0969]
tasks_100_speedup = [0.9897, 2.9044, 6.2222, 12.9968]
tasks_50_efficiency = [0.50, 0.7069, 0.6972, 0.6936]
tasks_100_efficiency = [0.4949, 0.7261, 0.7778, 0.8123]

# Plot Execution Time
plt.figure(figsize=(10, 6))
plt.plot(ranks, tasks_50_time, marker='o', label='50 Tasks')
plt.plot(ranks, tasks_100_time, marker='s', label='100 Tasks')
plt.xlabel("Ranks")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs Ranks")
plt.xticks(ranks)
plt.grid(True)
plt.legend()
plt.savefig("execution_time.png", dpi=300)
plt.show()

# Plot Speedup
plt.figure(figsize=(10, 6))
plt.plot(ranks, tasks_50_speedup, marker='o', label='50 Tasks')
plt.plot(ranks, tasks_100_speedup, marker='s', label='100 Tasks')
plt.xlabel("Ranks")
plt.ylabel("Speedup")
plt.title("Speedup vs Ranks")
plt.xticks(ranks)
plt.grid(True)
plt.legend()
plt.savefig("speedup.png", dpi=300)
plt.show()

# Plot Efficiency
plt.figure(figsize=(10, 6))
plt.plot(ranks, tasks_50_efficiency, marker='o', label='50 Tasks')
plt.plot(ranks, tasks_100_efficiency, marker='s', label='100 Tasks')
plt.xlabel("Ranks")
plt.ylabel("Efficiency")
plt.title("Efficiency vs Ranks")
plt.xticks(ranks)
plt.grid(True)
plt.legend()
plt.savefig("efficiency.png", dpi=300)
plt.show()
