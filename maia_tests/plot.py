import matplotlib.pyplot as plt

# Given dictionaries
harmonia_data = {'1': 0.448, '2': 0.533, '3': 0.507, '4': 0.549, '5': 0.535, '6': 0.523, '7': 0.533, '8': 0.523, '9': 0.511, '10': 0.513, '11': 0.566, '12': 0.542, '13': 0.552, '14': 0.54, '15': 0.538, '16': 0.547, '17': 0.552, '18': 0.538, '19': 0.567, '20': 0.583, '21': 0.539, '22': 0.549, '23': 0.562, '24': 0.553, '25': 0.544, '26': 0.556, '27': 0.56, '28': 0.543, '29': 0.534, '30': 0.564, '31': 0.584, '32': 0.541, '33': 0.579, '34': 0.579, '35': 0.591, '36': 0.582, '37': 0.574, '38': 0.584, '39': 0.568}
maia_data = {'1': 0.061, '2': 0.29, '3': 0.402, '4': 0.509, '5': 0.507, '6': 0.525, '7': 0.543, '8': 0.509, '9': 0.526, '10': 0.518, '11': 0.553, '12': 0.522, '13': 0.542, '14': 0.528, '15': 0.534, '16': 0.546, '17': 0.556, '18': 0.507, '19': 0.546, '20': 0.57, '21': 0.533, '22': 0.529, '23': 0.515, '24': 0.529, '25': 0.534, '26': 0.548, '27': 0.543, '28': 0.526, '29': 0.52, '30': 0.55, '31': 0.567, '32': 0.526, '33': 0.56, '34': 0.576, '35': 0.571, '36': 0.559, '37': 0.556, '38': 0.579, '39': 0.556}

# Convert keys to integers for proper sorting
x_values1 = sorted(map(int, harmonia_data.keys()))
y_values1 = [harmonia_data[str(x)] for x in x_values1]

x_values2 = sorted(map(int, maia_data.keys()))
y_values2 = [maia_data[str(x)] for x in x_values2]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(x_values1, y_values1, marker='o', linestyle='-', color='b', label='Harmonia Model')
plt.plot(x_values2, y_values2, marker='s', linestyle='--', color='r', label='Maia Model')

# Labels and title
plt.xlabel('Move Number')
plt.ylabel('Percentage correct')
plt.title('Move Prediction Accuracy Over Time')

# Show legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
