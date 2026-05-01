import numpy as np
import matplotlib.pyplot as plt

# --- Define the range for the parameter 't' ---
# np.linspace(start, end, number_of_points)
t = np.linspace(0, 2 * np.pi, 500)

# --- Define your parametric equations here ---
# You can change these functions to plot different curves.
def x_t(t):
    """Defines the x-component of the curve."""
    return 0.5 * np.cos(t)

def y_t(t):
    """Defines the y-component of the curve."""
    return 0.4 * np.sin(t) - 0.3 * np.sin(t)**2

# --- Plotting the curve ---
plt.figure(figsize=(8, 6)) # Create a figure for the plot
plt.plot(x_t(t), y_t(t))   # Plot x(t) vs y(t)
plt.plot((lambda t: 1.3 * np.cos(t))(t), (lambda t: np.sin(t))(t))   # Plot x(t) vs y(t)

# --- Customize the plot ---
plt.title('Parametric Curve: $x(t)$ vs $y(t)$') # Add a title
plt.xlabel('$x(t)$')                          # Add x-axis label
plt.ylabel('$y(t)$')                          # Add y-axis label
plt.grid(True)                                # Add a grid
plt.axis('equal')                             # Ensure aspect ratio is equal
plt.show()                                    # Display the plot
