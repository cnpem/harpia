import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise

from harpia import filters

image = data.gravel()
image = (image - image.min()) / (image.max() - image.min())
poisson_noisy_image = random_noise(image, mode="gaussian")

# Create a copy of the noisy image for filtering
filtered_image = poisson_noisy_image.copy()

# Apply anisotropic diffusion filter
filters.anisotropic_diffusion2D(
    filtered_image, total_iterations=1, delta_t=0.1, kappa=120, diffusion_option=3
)

# Save the filtered and original images using plt.imsave
plt.imsave("./filtered_image.png", filtered_image, cmap="gray")
plt.imsave("./original_image.png", poisson_noisy_image, cmap="gray")
