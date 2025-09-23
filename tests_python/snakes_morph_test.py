import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Import Harpia-backed functions as hp
import harpia as hp

# === Morphological ACWE ===
image = ski.util.img_as_float(ski.data.camera())

# Initial level set
init_ls = ski.segmentation.checkerboard_level_set(image.shape, 6).astype(bool)

# Harpia version
ls_hp = hp.segmentation.morphological_chan_vese(
    image.astype(np.float32),
    num_iter=35,
    init_level_set=init_ls,
    smoothing=3
)

# scikit-image version
ls_ski = ski.segmentation.morphological_chan_vese(
    image, num_iter=35, init_level_set=init_ls, smoothing=3
)

# === Morphological GAC ===
image2 = ski.util.img_as_float(ski.data.coins())
gimage = ski.segmentation.inverse_gaussian_gradient(image2)

# Initial level set
init_ls2 = np.zeros(image2.shape, dtype=bool)
init_ls2[10:-10, 10:-10] = True

# Harpia version
ls2_hp = hp.segmentation.morphological_geodesic_active_contour(
    gimage.astype(np.float32),
    num_iter=230,
    init_level_set=init_ls2,
    smoothing=1,
    balloon=-1.0,
    threshold=0.69,
)

# scikit-image version
ls2_ski = ski.segmentation.morphological_geodesic_active_contour(
    gimage,
    num_iter=230,
    init_level_set=init_ls2,
    smoothing=1,
    balloon=-1,
    threshold=0.69,
)

# === Plot results: 2 rows × 4 cols ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
ax = axes.flatten()

# ACWE Harpia
ax[0].imshow(image, cmap="gray")
ax[0].contour(ls_hp, [0.5], colors="r")
ax[0].set_axis_off()
ax[0].set_title("ACWE segmentation (hp)", fontsize=12)

ax[1].imshow(ls_hp, cmap="gray")
ax[1].set_axis_off()
ax[1].set_title("ACWE mask (hp)", fontsize=12)

# ACWE scikit-image
ax[2].imshow(image, cmap="gray")
ax[2].contour(ls_ski, [0.5], colors="r")
ax[2].set_axis_off()
ax[2].set_title("ACWE segmentation (ski)", fontsize=12)

ax[3].imshow(ls_ski, cmap="gray")
ax[3].set_axis_off()
ax[3].set_title("ACWE mask (ski)", fontsize=12)

# GAC Harpia
ax[4].imshow(image2, cmap="gray")
ax[4].contour(ls2_hp, [0.5], colors="r")
ax[4].set_axis_off()
ax[4].set_title("GAC segmentation (hp)", fontsize=12)

ax[5].imshow(ls2_hp, cmap="gray")
ax[5].set_axis_off()
ax[5].set_title("GAC mask (hp)", fontsize=12)

# GAC scikit-image
ax[6].imshow(image2, cmap="gray")
ax[6].contour(ls2_ski, [0.5], colors="r")
ax[6].set_axis_off()
ax[6].set_title("GAC segmentation (ski)", fontsize=12)

ax[7].imshow(ls2_ski, cmap="gray")
ax[7].set_axis_off()
ax[7].set_title("GAC mask (ski)", fontsize=12)

fig.tight_layout()

# Save comparison figure
plt.savefig("morph_snakes_hp_vs_ski.png", dpi=300, bbox_inches="tight")
plt.close(fig)
