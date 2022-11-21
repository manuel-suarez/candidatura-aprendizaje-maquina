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
x_files  = glob(os.path.join(local_dir, 'RGB', '*.jpg'))
y_files  = glob(os.path.join(local_dir, 'haze', 'beta05', '*.jpg'))
x_files.sort()
y_files.sort()
x_files=np.array(x_files)
y_files=np.array(y_files)

# Verificamos las primeras 5 entradas para identificar si corresponden al mismo escenario
for xfile, yfile in zip(x_files[:5], y_files[:5]):
  print(xfile, yfile)

# Comparamos longitud de conjunto de entrenamiento
print(len(x_files), len(y_files))

# Configuración del paso de entrenamiento
BUFFER_SIZE      = len(y_files)
steps_per_epoch  = BUFFER_SIZE // BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch)

# Función de lectura de imágnes
def read_and_decode(file):
  '''
  Lee, decodifica y redimensiona la imagen.
  Aplica aumentación
  '''
  # Lectura y decodificación
  img = tf.io.read_file(file)
  img = tf.image.decode_png(img)
  img = tf.cast(img, tf.float32)
  # Normalización
  img = img / 127.5 - 1
  # Redimensionamiento
  img = tf.image.resize(img, INPUT_DIM[:2],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return img


def load_images(x_file, y_file, flip=True):
  '''
  Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado

  Aumentación: Flip horizontal aleatorio, sincronizado
  '''
  x_img = read_and_decode(x_file)
  y_img = read_and_decode(y_file)
  # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
  if flip and tf.random.uniform(()) > 0.5:
    x_img = tf.image.flip_left_right(x_img)
    y_img = tf.image.flip_left_right(y_img)

  return x_img, y_img


def display_images(figname, x_imgs=None, y_imgs=None, rows=3, offset=0):
  '''
  Despliega conjunto de imágenes izquierda y derecha junto a la disparidad
  '''
  # plt.figure(figsize=(20,rows*2.5))
  fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 2.5))
  for i in range(rows):
    ax[i, 0].imshow((x_imgs[i + offset] + 1) / 2)
    ax[i, 0].set_title('Image')
    ax[i, 1].imshow((y_imgs[i + offset] + 1) / 2)
    ax[i, 1].set_title('Haze')

  plt.tight_layout()
  plt.savefig(figname)

x_imgs = []
y_imgs  = []

# Cargamos 3 imagenes
for i in range(3):
    x_img, y_img = load_images(x_files[i], y_files[i])
    x_imgs.append(x_img)
    y_imgs.append(y_img)
# Verificamos la forma de las imagenes cargadas
print(x_imgs[0].shape, y_imgs[0].shape)

# Verificamos
display_images("figura1.png", x_imgs, y_imgs, rows=3)

# Definición de los conjuntos de entrenamiento y prueba
idx = int(BUFFER_SIZE*.8)

train_x = tf.data.Dataset.list_files(x_files[:idx], shuffle=False)
train_y = tf.data.Dataset.list_files(y_files[:idx], shuffle=False)

test_x = tf.data.Dataset.list_files(x_files[idx:], shuffle=False)
test_y = tf.data.Dataset.list_files(y_files[idx:], shuffle=False)

train_xy = tf.data.Dataset.zip((train_x, train_y))
train_xy = train_xy.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
train_xy = train_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_xy = train_xy.batch(BATCH_SIZE)

test_xy = tf.data.Dataset.zip((test_x, test_y))
test_xy = test_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_xy = test_xy.batch(BATCH_SIZE)

# Verificación de carga de archivos por medio de la definición de los Dataset
for x, y in train_xy.take(3):
    display_images("figura2.png", x, y, rows=3)
    break
for x, y in test_xy.take(3):
    display_images("figura3.png", x, y, rows=3)
    break
