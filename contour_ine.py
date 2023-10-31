import numpy as np
import matplotlib.pyplot as plt
from rms import RMSprop

# Parameters for the bivariate normal distribution
mu = [0, 0]  # Mean
cov = [[1, 0.25], [0.25, 0.5]]  # Covariance matrix

# Generate data points for the contour plot
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)
Z = np.dstack((X, Y))
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X
# pos[:, :, 1] = Y

# Calculate the probability density function (PDF) for the bivariate normal distribution
Z_pdf = np.exp(-0.5 * np.einsum('...k,kl,...l->...', Z - mu, np.linalg.inv(cov), Z - mu))
Z_pdf /= 2 * np.pi * np.sqrt(np.linalg.det(cov))

fig, ax = plt.subplots()

# Plot the contour lines
ax.contour(X, Y, -np.log(Z_pdf), levels=4, cmap='Greens')

# Add labels and title
# plt.text(-0.1, 0.5, 'Y', rotation=0, va='center', ha='center', transform=plt.gca().transAxes)
ax.set_title('Contour Plot of Negative Log Bivariate Normal Distribution')

# Set the region of Y and X to be displayed
ax.set_xlim(-40, 40)
ax.set_ylim(-30, 30)

# Set the aspect ratio to be equal
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Define the gradient used in RMSprop
def column(x,i):
    return [row[i] for row in x]
# return the list of gradients
def G(X):
    g = []
    for i in range(len(X)):
        gradient = 0
        cov_row = cov[i] # cov is defined at the top
        cov_col = column(cov,i)
        for j, x in enumerate(X):
            gradient += x * (cov_row[j] + cov_col[j])
        g.append(gradient * 0.5)
    return g

def draw_iteration(n, decay):
    prop = RMSprop(G, [-30,-10])
    prop.decay = decay
    prop.learn = 0.75

    prop.start(n)
    # Extract x and y coordinates from the points
    x_coords = [x[0] for x in prop.result_x]
    y_coords = [x[1] for x in prop.result_x]
    # Plot the points
    ax.plot(x_coords, y_coords, '-', label='# iterate = %d; decay = %.1f'%(n,decay))

draw_iteration(100, 0.9)
# Display the plot
plt.legend()
plt.show()