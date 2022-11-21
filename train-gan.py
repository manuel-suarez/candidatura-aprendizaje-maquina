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

# Definición de variables
INPUT_DIM     = (256,256,3) # Tamaño de entrada
OUTPUT_CHANNELS = 3         # La salida es la imágen RGB limpia
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0