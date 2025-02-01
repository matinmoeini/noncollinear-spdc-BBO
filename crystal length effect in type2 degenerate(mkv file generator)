import numpy as np
import matplotlib.pyplot as plt
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio_ffmpeg as ffmpeg
from PIL import Image
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap

# Constants
landapump = 0.405
landa1type2 = 0.810
landa2type2 = 1 / ((1 / landapump) - (1 / landa1type2))
l = 1e-3
threshold = 0  # Define an example threshold for masking

# Define ranges for phi and theta
phi_values = np.linspace(0, 2 * np.pi, 100)  # X-axis values
theta_values = np.linspace(0, 7, 100) * np.pi / 180  # Y-axis values
theta_fixed = 41.9 * (np.pi / 180)  # Initial theta value

# Precompute refractive indices
nepump = np.sqrt(2.3730 + (0.0128 / (landapump ** 2 - 0.0156)) - (0.0044 * (landapump ** 2)))
nopump = np.sqrt(2.7405 + (0.0184 / (landapump ** 2 - 0.0179)) - (0.0155 * (landapump ** 2)))

# Prepare figure and axis
fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))
ax.set_theta_offset(np.pi / 2)  # Rotate by 90 degrees (pi/2 radians)
# Create a custom colormap from white to yellow
colors = [(1, 1, 1), (202/255, 116/255, 57/255)]  # RGB values for white and yellow
n_bins = 100  # Number of bins
cmap_name = 'white_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Define the custom colors (white to #ca7439)
colors2 = [(1, 1, 1), (202/255, 116/255, 57/255)]  # RGB for white and #ca7439
n_bins2 = 100  # Number of bins
cmap_name2 = 'white_to_orange'
cm2 = LinearSegmentedColormap.from_list(cmap_name2, colors2, N=n_bins2)

# Update function for animation
def update(frame):
    global l
    ax.clear()  # Clear the axis for the next frame

    l += 0.04e-3   # Increment theta_fixed

    nethetapump_value = np.sqrt(1 / ((np.sin(theta_fixed)**2 / nepump**2) + (np.cos(theta_fixed)**2 / nopump**2)))
    phi, theta = np.meshgrid(phi_values, theta_values)

    ne1type2 = np.sqrt(2.3730 + (0.0128 / (landa1type2 ** 2 - 0.0156)) - (0.0044 * (landa1type2 ** 2)))
    no1type2 = np.sqrt(2.7405 + (0.0184 / (landa1type2 ** 2 - 0.0179)) - (0.0155 * (landa1type2 ** 2)))

    thetai = np.arccos(np.sin(theta_fixed) * np.sin(theta) * np.cos(phi) + np.cos(theta_fixed) * np.cos(theta))
    netheta1type2 = np.sqrt(1 / ((np.sin(thetai)**2 / ne1type2**2) + (np.cos(thetai)**2 / no1type2**2)))

    no2type2 = np.sqrt(2.7405 + (0.0184 / (landa2type2 ** 2 - 0.0179)) - (0.0155 * (landa2type2 ** 2)))

    value = landa2type2 * netheta1type2 * np.sin(theta) / (landa1type2 * no2type2)
    value = np.clip(value, -1, 1)

    thetaps = np.arcsin(value)

    eq2type2 = (nethetapump_value / (1e-6 * landapump)) - (no2type2 * np.sqrt(1 - value**2) / (1e-6 * landa2type2)) - \
               (netheta1type2 * np.cos(theta) / (1e-6 * landa1type2))

    intensity = (l * np.sinc(eq2type2 * l / 2))**2
    masked_intensity = np.ma.masked_where((intensity >= 0) & (intensity <= threshold), intensity)

    arrphinoncollineartype2 = phi
    arrthetanoncollineartype2 = np.arcsin(netheta1type2 * np.sin(theta)) * 180 / np.pi
    arrphinoncollineartype2minus = phi + np.pi
    arrthetanoncollineartype2minus = np.arcsin(no2type2 * np.sin(thetaps)) * 180 / np.pi

    norm1 = Normalize(vmin=threshold, vmax=np.nanmax(intensity))

    # Plot updated heatmaps
    ax.pcolormesh(arrphinoncollineartype2, arrthetanoncollineartype2, masked_intensity, shading='auto', cmap=cm2, norm=norm1, alpha=0.8)
    ax.pcolormesh(arrphinoncollineartype2minus, arrthetanoncollineartype2minus, masked_intensity, shading='auto', cmap=cm, norm=norm1, alpha=0.4)
    ax.set_title(f'Type 2 Noncollinear degenerate Phase Matching_length\nlength {(l*1e3):.2f}mm',fontsize=16)
    ax.set_theta_offset(np.pi / 2)  # Rotate by 90 degrees (pi/2 radians)
# Set up the video writer with imageio-ffmpeg
fps = 20
filename = 'type2_noncollinear_degenerate_length.mkv'

# Create the writer using the appropriate function from imageio-ffmpeg
with imageio.get_writer(filename, mode='I', fps=fps) as writer:
    for frame in range(100):  # 100 frames for the animation
        update(frame)  # Update the plot for each frame
        
        # Convert the figure to a PIL Image
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        
        # Convert the PIL Image to a numpy array and append it to the video writer
        writer.append_data(np.array(img))

print(f"Movie saved as {filename}")
