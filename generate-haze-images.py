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