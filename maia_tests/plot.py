import matplotlib.pyplot as plt

# Given dictionary
data = {'1': 0.07, '2': 0.32, '3': 0.51, '4': 0.46, '5': 0.56, '6': 0.57, '7': 0.54, '8': 0.57, '9': 0.51, '10': 0.6, '11': 0.42, '12': 0.56, '13': 0.52, '14': 0.58, '15': 0.5, '16': 0.49, '17': 0.5, '18': 0.58, '19': 0.5, '20': 0.53, '21': 0.41, '22': 0.47, '23': 0.53, '24': 0.56, '25': 0.51, '26': 0.6, '27': 0.58, '28': 0.64, '29': 0.57, '30': 0.43, '31': 0.44, '32': 0.55, '33': 0.51, '34': 0.56, '35': 0.59, '36': 0.46, '37': 0.6, '38': 0.51, '39': 0.59}

# Convert keys to integers for proper sorting
x_values = sorted(map(int, data.keys()))
y_values = [data[str(x)] for x in x_values]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Values')

# Labels and title
plt.xlabel('Move Number')
plt.ylabel('Percentage correct')
plt.title('Harmonia Move prediction accuracy as a function of time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
