import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the simulation
num_trajectories = 5  # Number of trajectories to plot
num_steps = 1000  # Number of steps in each trajectory
start_point = np.array([-2.5, 0.0])  # Starting point (z_T)
end_point = np.array([2.5, 0.0])  # Ending point (z_0)
diffusion_rate = 0.0058  # Controls the "randomness" of the walk (reduced for smoother lines)

# Create a figure and axes for the plot
plt.figure(figsize=(10, 6))
# plt.title('Latent Diffusion Bridge', fontsize=16, fontweight='bold')

# Loop to generate and plot each trajectory
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(num_trajectories):
    # Initialize the current position
    current_position = np.copy(start_point)
    trajectory = [current_position]

    for t in range(num_steps):
        # Calculate the drift towards the end point
        drift = (end_point - current_position) / (num_steps - t)

        # Calculate the random walk component (diffusion)
        # We only apply diffusion on the Y-axis to make it a more horizontal plot
        diffusion = np.array([0, np.random.randn() * diffusion_rate])

        # Update the position
        current_position = current_position + drift + diffusion
        trajectory.append(current_position)

    # Convert the trajectory list to a numpy array for easy plotting
    trajectory = np.array(trajectory)

    # Plot the trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i % len(colors)], linewidth=2)

# Annotate the start and end points
plt.text(start_point[0], start_point[1], r'$z_0$', fontsize=20, ha='right', va='center')
# Create a simple box for the z0 annotation
# plt.gca().add_patch(plt.Rectangle((start_point[0] - 0.2, start_point[1] - 0.5), 0.1, 1,
#                                   fill=False, edgecolor='black', linewidth=1.5))
plt.text(end_point[0], end_point[1], r'$z_T$', fontsize=20, ha='left', va='center')

# Adjust plot appearance to be more like the original image
plt.axis('off')  # Hide the axes for a cleaner look
plt.tight_layout()

# Save the plot to a file
plt.savefig('diffusion_bridge_smoother.png', dpi=300)

plt.show()