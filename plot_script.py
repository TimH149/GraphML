import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.size'] = 15

list1 = [0.239, 0.438, 0.592, 0.856, 1.00]
list2 = [0.249, 0.477, 0.591, 0.820, 0.99]

palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
palette = np.random.choice(palettes)

# Get two different colors from the chosen palette
colors = sns.color_palette(palette, 2)

# Create a line plot with list1 and list2
sns.lineplot(x=range(len(list1)), y=list1, color=colors[0], label='True Homophily', marker='o', markersize=10)
sns.lineplot(x=range(len(list2)), y=list2, color=colors[1], label='Predicted Homophily', marker='o', markersize=10)

# Annotate the points with their values
for i, value in enumerate(list1):
    plt.text(i, value+ 0.05, str(value), color=colors[0])
for i, value in enumerate(list2):
    plt.text(i, value-0.05, str(value), color=colors[1])

# Add a legend with custom names
plt.legend()

# Add labels for the x-axis and y-axis
plt.xlabel('Graph Index')
plt.ylabel('Label Homophily')

# Show the plot
plt.savefig("plot/homophily.png")