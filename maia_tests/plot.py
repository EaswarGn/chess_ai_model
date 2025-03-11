import matplotlib.pyplot as plt

# Given dictionary
data = {'1': 0.52, '2': 0.61, '3': 0.53, '4': 0.53, '5': 0.48, '6': 0.51, '7': 0.51, '8': 0.6, '9': 0.56, '10': 0.56, '11': 0.57, '12': 0.57, '13': 0.52, '14': 0.52, '15': 0.57, '16': 0.59, '17': 0.51, '18': 0.45, '19': 0.52, '20': 0.53, '21': 0.64, '22': 0.54, '23': 0.54, '24': 0.53, '25': 0.62, '26': 0.64, '27': 0.69, '28': 0.7, '29': 0.57, '30': 0.57, '31': 0.61, '32': 0.68, '33': 0.58, '34': 0.51, '35': 0.66, '36': 0.64, '37': 0.62, '38': 0.64, '39': 0.59}

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
