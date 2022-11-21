# Imports
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import datetime
from IPython import display


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Definición de parámetros
INPUT_DIM     = (256,256,3) # Tamaño de entrada
OUTPUT_CHANNELS = 3         # La salida es la imágen RGB limpia
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0

# Definición y configuración de las rutas
local_dir = '/home/est_posgrado_manuel.suarez/data/ReDWeb-S/trainset'
xfiles  = glob(os.path.join(local_dir, 'RGB', '*.jpg'))
yfiles  = glob(os.path.join(local_dir, 'haze', 'beta05', '*.png'))
xfiles.sort()
yfiles.sort()
xfiles=np.array(xfiles)
yfiles=np.array(yfiles)

# Verificamos las primeras 5 entradas para identificar si corresponden al mismo escenario
for xl, xr, y in zip(xl_files[:5], xr_files[:5], y_files[:5]):
  print(xl, xr, y)