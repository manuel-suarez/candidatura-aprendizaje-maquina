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

# Definición de la arquitectura de la red

# Bloque downsample
def downsample(filters, size, apply_batchnorm=True):
    '''
    Bloque de codificación (down-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters           = filters,
                                      kernel_size       = size,
                                      strides           = 2,
                                      padding           = 'same',
                                      kernel_initializer= initializer,
                                      use_bias          = False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Aplicación downsample
down_model  = downsample(3, 4)
down_result = down_model(tf.expand_dims(x_img, 0))
print(down_result.shape)

# Bloque upsample
def upsample(filters, size, apply_dropout=False):
    '''
    Bloque de decodicación (up-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters           = filters,
                                               kernel_size       = size,
                                               strides           = 2,
                                               padding           = 'same',
                                               kernel_initializer= initializer,
                                               use_bias          = False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# Aplicación upsample
up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

# Generador
def Generator():
    '''
    UNet
    '''
    # En relación con el planteamiento del problema tenemos dos imágenes
    # (izquiera y derecha) que consisten el input de entrada por lo que debemos
    # definir dos capas de entrada que posteriormente uniremos por una capa de
    # concatenación
    x_input = tf.keras.layers.Input(shape=INPUT_DIM)
    # Definimos además los bloques down y up que componen la arquitectura de la
    # UNet
    down_stack = [
        downsample(64,  4, apply_batchnorm=False),# (batch_size, 128, 128, 64)
        downsample(128, 4),                       # (batch_size, 64,  64,  128)
        downsample(256, 4),                       # (batch_size, 32,  32,  256)
        downsample(512, 4),                       # (batch_size, 16,  16,  512)
        downsample(512, 4),                       # (batch_size, 8,   8,   512)
        downsample(512, 4),                       # (batch_size, 4,   4,   512)
        downsample(512, 4),                       # (batch_size, 2,   2,   512)
        downsample(512, 4),                       # (batch_size, 1,   1,   512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),     # (batch_size, 2,    2,  1024)
        upsample(512, 4, apply_dropout=True),     # (batch_size, 4,    4,  1024)
        upsample(512, 4, apply_dropout=True),     # (batch_size, 8,    8,  1024)
        upsample(512, 4),                         # (batch_size, 16,   16, 1024)
        upsample(256, 4),                         # (batch_size, 32,   32, 512)
        upsample(128, 4),                         # (batch_size, 64,   64, 256)
        upsample(64,  4),                         # (batch_size, 128, 128, 128)
    ]

    # Definimos la capa de salida (estimación del campo de disparidades) en
    # función del número de canales definidos previamente
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    # Definición de la arquitectura de la red
    x = x_input
    # Codificador
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)  # se agrega a una lista la salida cada vez que se desciende en el generador
    skips = reversed(skips[:-1])
    # Decodificador (bloques upsample y skip connections)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    # Capa de salida
    x = last(x)
    # Construcción del modelo
    # Se requieren dos inputs y proporciona una única salida
    return tf.keras.Model(inputs=x_input, outputs=x)
# Estructura del generador
generator = Generator()
generator.summary()
# Verificación de la salida del generador
gen_output = generator(x_img[tf.newaxis, ...], training=False)
plt.figure()
plt.imshow(gen_output[0, ...][:,:,0]*50)
plt.savefig("figura4.png")

# Discriminador
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp_l = tf.keras.layers.Input(shape=[256, 256, 3], name='input_left_image')
    inp_r = tf.keras.layers.Input(shape=[256, 256, 3], name='input_right_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')
    x = tf.keras.layers.concatenate(
        [inp_l, inp_r, tar])  # (batch_size, 256, 256, 3 channels left + 3 channels right + disparity channel)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(filters=1,
                                  kernel_size=4,
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp_l, inp_r, tar], outputs=last)
# Resumen de la arquitectura del discriminador
discriminator = Discriminator()
discriminator.summary()