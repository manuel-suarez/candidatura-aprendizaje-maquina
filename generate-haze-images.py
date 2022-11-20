# Configuración inicial
import os
import numpy as np
from PIL import Image
# from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Configuración de rutas
# Ruta en el clúster
root_folder = "/home/est_posgrado_manuel.suarez/data"
img_dir = "ReDWeb-S"                # Base de datos
base_dir = ["trainset", "testset"]
clean_dir = "RGB"
depth_dir = "depth"
target_dir = "haze"

# Visualización y ejemplificación de "hazing"
fig, axs = plt.subplots(3, 3, figsize=(8, 12))
beta = 2.5  # Scattering coefficient
A = 1  # Atmospheric light
for i, filename in enumerate(os.listdir(os.path.join(root_folder, img_dir, base_dir[0], clean_dir))[:3]):
    # Image
    img = np.array(Image.open(os.path.join(root_folder, img_dir, base_dir[0], clean_dir, filename)))
    # Depth map
    depthname = filename.split(".")[0] + ".png"
    depth = np.array(Image.open(os.path.join(root_folder, img_dir, base_dir[0], depth_dir, depthname)))
    # Hazing
    tx = np.exp(-beta * depth / 255)  # Transmission map
    haze = np.zeros_like(img, dtype=np.uint8)
    haze[:, :, 0] = img[:, :, 0] * tx + A * 255 * (1 - tx)
    haze[:, :, 1] = img[:, :, 1] * tx + A * 255 * (1 - tx)
    haze[:, :, 2] = img[:, :, 2] * tx + A * 255 * (1 - tx)
    # Max-min
    print(np.max(img), np.min(img))
    print(np.max(depth), np.min(depth))
    print(np.max(haze), np.min(haze))
    # Plotting
    axs[i, 0].imshow(img)
    axs[i, 1].imshow(depth)
    axs[i, 2].imshow(haze)
plt.show()
