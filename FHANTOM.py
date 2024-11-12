# Imports
import numpy as np
from scipy.special import lpmv
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import time

# Constants
RES = 256
NUM_ANGLES = 4 # needs to be same as m (this is 4 for now)
DISTRIBUTION = 'Y22_Y42'

# Functions
def distribution_coefficient(m, l):
    return np.sqrt((2 * l + 1) / (4 * np.pi) * np.math.factorial(l - m) / np.math.factorial(l + m))

def gaussian(mu, sd, x):
    return np.exp(-0.5 * (x - mu)**2 / sd**2)

def generate_distribution_old(res):
    # Distribution Constants
    mu = 0.7
    sd = 0.05

    # Generate Coordinates
    x = np.linspace(-1, 1, res)
    x, y, z = np.meshgrid(x, x, x)

    # Generate Polar Coordinates
    R = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / R
    cos_theta[np.isnan(cos_theta)] = 1

    # Generate Distribution
    lp2 = 0.5 * (3 * cos_theta**2 - 1)
    I = np.exp(-0.5 * (R - mu)**2 / sd**2) * (1 + 2 * lp2)
    return I / np.max(I)

    # Next distribution to try: I = Gauss(.3) \times Y_{22} + Guass(.7) \times Y_{42}

def generate_distribution(res):
    # Distribution Constants
    mu = .5
    sd = 0.05

    # Generate Coordinates
    x = np.linspace(-1, 1, res)
    x, y, z = np.meshgrid(x, x, x)

    # Generate Polar Coordinates
    R = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = np.nan_to_num(z / R, nan=1.0)
    cos_phi = np.nan_to_num(y / np.sqrt(x**2 + y**2), nan=1.0)
    cos_2phi = 2 * cos_phi**2 - 1
    cos_4phi = 2 * cos_2phi**2 - 1

    # Generate Distribution
    I = -1 * gaussian(mu, sd, R) * distribution_coefficient(2, 4) * lpmv(2, 4, cos_theta) * cos_2phi
    return I / np.max(I)

def generate_rotations(I, angles):
    # Create an array of evenly spaced angles between 0 and 90
    I_proj_rot = np.zeros((len(angles), RES, RES))

    # Rotate the distribution about the y-axis for each angle
    for i, angle in enumerate(angles):
        I_rot = rotate(I, angle, axes=(0, 2), reshape=False)
        I_proj_rot[i] = np.sum(I_rot, axis=0) # Sum over the x-axis to project onto the y-z plane
    return I_proj_rot

def plot_projections(I, angles, plot_title, file_name):
    grid_size = int(np.ceil(np.sqrt(len(angles)))) # Determine the grid size for subplots

    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for ax, proj, angle in zip(axes, I, angles):
        ax.imshow(proj, cmap='viridis', extent=(-1, 1, -1, 1))
        ax.set_title(f'{plot_title}: {angle:.1f}°')
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
        

    # Hide any unused subplots
    for ax in axes[len(angles):]:
        ax.axis('off')

    # Save the plot
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def fourier_transform_angles(I):
    # Reshape the distribution for the Fourier Transform
    I_ft = np.transpose(I, (1, 2, 0)) # Permute list of matrices to matrix of lists
    I_ft = np.fft.fft(I_ft, axis=2)
    return np.transpose(I_ft, (2, 0, 1)) # Permute matrix of fourier transformed lists back to list of matrices

def fourier_transform_plots(I_ft, angles, component, file_name):
    grid_size = int(np.ceil(np.sqrt(len(angles))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot the Fourier Transform for each angle
    for ax, ft_proj, angle in zip(axes, I_ft, angles):
        if component == 'magnitude':
            data = np.abs(ft_proj)
        elif component == 'real':
            data = np.real(ft_proj)
        ax.imshow(data, cmap='viridis', extent=(-1, 1, -1, 1))
        ax.set_title(f'{component.capitalize()} FT for Rotation: {angle:.1f}°')
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
    
    # Hide any unused subplots
    for ax in axes[len(angles):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{file_name}_{component}.png')
    plt.close()

# Start Timer
start_time = time.time()

I = generate_distribution(RES)
angles = np.linspace(0, 90, NUM_ANGLES)
I_proj_rot = generate_rotations(I, angles)
plot_projections(I_proj_rot, angles, 'Rotation', f'{NUM_ANGLES}_rotations_0to90_{DISTRIBUTION}')
I_ft_mat = fourier_transform_angles(I_proj_rot)
fourier_transform_plots(I_ft_mat, angles, 'magnitude', f'{NUM_ANGLES}_rotations_0to90_FT_{DISTRIBUTION}')
fourier_transform_plots(I_ft_mat, angles, 'real', f'{NUM_ANGLES}_rotations_0to90_FT_{DISTRIBUTION}')

# Stop Timer
print(f'Elapsed Time: {time.time() - start_time:.2f} seconds')