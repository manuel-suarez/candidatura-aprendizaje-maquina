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

# Preparamos directorios donde pondremos los diferentes niveles de hazing
for dirname in base_dir:
    if not os.path.isdir(os.path.join(root_folder,img_dir,dirname,'haze')):
        os.mkdir(os.path.join(root_folder,img_dir,dirname,'haze'))
        os.mkdir(os.path.join(root_folder,img_dir,dirname,'haze','beta05'))
        os.mkdir(os.path.join(root_folder,img_dir,dirname,'haze','beta15'))
        os.mkdir(os.path.join(root_folder,img_dir,dirname,'haze','beta25'))

# Generación de imágenes sintéticas

# Tres valores para el coeficiente de difusión
betas = [0.5, 1.5, 2.5] # Scattering coefficient
dirbetas = ['beta05', 'beta15', 'beta25']
A = 1      # Atmospheric light
for beta, dirbeta in zip(betas, dirbetas):
    for basedir in base_dir:
        for i, filename in enumerate(os.listdir(os.path.join(root_folder,img_dir,basedir,clean_dir))):
            # Image
            img = np.array(Image.open(os.path.join(root_folder,img_dir,basedir,clean_dir,filename)))
            # Depth map
            depthname = filename.split(".")[0]+".png"
            depth = np.array(Image.open(os.path.join(root_folder,img_dir,basedir,depth_dir,depthname)))
            # Verificamos coincidencia de dimensiones entre la imagen y su mapa de profundidad (recortamos si es necesario)
            if img.shape[0] != depth.shape[0] or img.shape[1] != depth.shape[1]:
                img = img[:depth.shape[0],:depth.shape[1],:]
            # Hazing
            tx = np.exp(-beta * depth / 255) # Transmission map
            haze = np.zeros_like(img, dtype=np.uint8)
            haze[:,:,0] = img[:,:,0]*tx + A*255*(1-tx)
            haze[:,:,1] = img[:,:,1]*tx + A*255*(1-tx)
            haze[:,:,2] = img[:,:,2]*tx + A*255*(1-tx)
            # Guardamos imagen con niebla
            img = Image.fromarray(np.clip(np.uint8(haze), 0, 255))
            img.save(os.path.join(root_folder,img_dir,basedir,'haze',dirbeta,filename))
print("Done!")

# Visualización de los niveles de hazing
fig, axs = plt.subplots(3, 5, figsize=(8, 12))
for i, filename in enumerate(os.listdir(os.path.join(root_folder,img_dir,base_dir[0],clean_dir))[:3]):
    # Image
    img = np.array(Image.open(os.path.join(root_folder,img_dir,base_dir[0],clean_dir,filename)))
    # Depth map
    depthname = filename.split(".")[0]+".png"
    depth = np.array(Image.open(os.path.join(root_folder,img_dir,base_dir[0],depth_dir,depthname)))
    # Haze image
    haze_b05 = np.array(Image.open(os.path.join(root_folder,img_dir,base_dir[0],'haze',dirbetas[0],filename)))
    haze_b15 = np.array(Image.open(os.path.join(root_folder,img_dir,base_dir[0],'haze',dirbetas[1],filename)))
    haze_b25 = np.array(Image.open(os.path.join(root_folder,img_dir,base_dir[0],'haze',dirbetas[2],filename)))
    # Max-min
    print(np.max(img), np.min(img))
    print(np.max(depth), np.min(depth))
    print(np.max(haze), np.min(haze))
    # Plotting
    axs[i, 0].imshow(img)
    axs[i, 1].imshow(depth)
    axs[i, 2].imshow(haze_b05)
    axs[i, 3].imshow(haze_b15)
    axs[i, 4].imshow(haze_b25)
plt.axis("off")
plt.show()