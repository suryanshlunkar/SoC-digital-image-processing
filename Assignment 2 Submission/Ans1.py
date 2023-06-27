import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load .mat file
# For part a use 'data/points2D_Set1.mat'
file = h5py.File('data/points2D_Set2.mat', 'r')

# Extract X and Y values from the file
X = file['x'][:]
Y = file['y'][:]

# Flatten the X and Y arrays
X = X.flatten()
Y = Y.flatten()

# Plot scatter plot of the points
plt.scatter(X, Y)

# Fit a line to the data using linear regression
coefficients = np.polyfit(X, Y, 1)
line = np.poly1d(coefficients)

# Generate points on the line for plotting
x_range = np.linspace(X.min(), X.max(), 100)
y_range = line(x_range)

# Overlay the line on the scatter plot
plt.plot(x_range, y_range, color='red')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Linear Regression Line')

# Display the plot
plt.show()
