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
x_files  = glob(os.path.join(local_dir, 'haze', 'beta25', '*.jpg'))
y_files  = glob(os.path.join(local_dir, 'RGB', '*.jpg'))
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

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_left_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate(
        [inp, tar])  # (batch_size, 256, 256, 3 channels left + 3 channels right + disparity channel)

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

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
# Resumen de la arquitectura del discriminador
discriminator = Discriminator()
discriminator.summary()
# Verificación del resultado del discriminador
disc_out = discriminator([x_img[tf.newaxis, ...], gen_output], training=False)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow((x_img+1)/2)
ax[1].imshow(disc_out[0, ..., -1]*200, vmin=-20, vmax=20, cmap='RdBu_r')  #*100
plt.tight_layout()
plt.savefig("figura5.png")

# Funciones de costo (usar MSE y MAE)
#loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#loss_object = tf.keras.losses.MeanSquaredError(reduction="auto")
loss_object = tf.keras.losses.MeanAbsoluteError(reduction="auto")
# Discriminador
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss        = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss   = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss  = real_loss + generated_loss
    return total_disc_loss
# Parámetro de ajuste
LAMBDA = 100
# Generador
def generator_loss(disc_generated_output, gen_output, target):
    '''
    el generador debe entrenarse para maximizar los errores de detección de imágenes sintéticas
    '''
    # Entropia cruzada a partir de logits
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Media de los Errores Absolutos
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
# Optimizadores
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# Directorio de checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# Función auxiliar para la visualización de resultados parciales
def generate_images(figname, model, x_input, y_input):
    '''
    Con training=True se obtienen las metricas sobre el Lote.
    En otro caso, no se evaluan y se regresan las del entrenamiento.
    '''
    y_pred = model([x_input], training=True)

    plt.figure(figsize=(15, 15))
    display_list = [y_input[0], x_input[0], y_pred[0]]
    title = ['Original $y$', 'Imágen con niebla $x$', 'Resultado p2p  $x^\prime$']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        if i < 3:
            plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')
    plt.savefig(figname)
# Verificación de función auxiliar
for x_input, y_input in train_xy.take(1):
    generate_images("figura6.png", generator, x_input, y_input)
    print(x_input.shape, y_input.shape)
    break
# Bitácora del entrenamiento
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Entrenamiento

@tf.function
def train_step(input_image, target, test_input_image, test_target, step):
    '''
    Cálculos realizados durante un paso del entrenamiento

    Dadas los pares (xl, xr), y (par de imagen izquierda-derecha estereoscópicas, campo de disparidades real):
    - Generar campo de disparidades sintéticos con UNet
    - Evaluar el discriminador con los campos de disparidades real y generado
    - Evaluar los costos del generador y del discriminador
    - Calcular los gradientes
    - Realiza los pasos de optimización
    - Registro de métricas
    '''

    # Usamos dos cintas de gradiente para el registro de las operaciones
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generamos campo de disparidades sintético
        gen_output = generator(input_image, training=True)
        # Obtenemos salida del discriminador con el campo real y sintético de
        # disparidades
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        # Obtenemos evaluación de las funciones de costo para el generador y el
        # discriminador
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Obtenemos gradientes a partir de las cintas
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Aplicamos a los modelos el paso de optimización
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # Usamos dos cintas de gradiente para el registro de las operaciones en los datos de prueba (para validación)
    with tf.GradientTape() as test_gen_tape, tf.GradientTape() as test_disc_tape:
        # Generamos campo de disparidades sintético
        test_gen_output = generator(test_input_image, training=True)
        # Obtenemos salida del discriminador con el campo real y sintético de
        # disparidades
        test_disc_real_output = discriminator([test_input_image, test_target], training=True)
        test_disc_generated_output = discriminator([test_input_image, test_gen_output], training=True)
        # Obtenemos evaluación de las funciones de costo para el generador y el
        # discriminador
        test_gen_total_loss, test_gen_gan_loss, test_gen_l1_loss = generator_loss(test_disc_generated_output,
                                                                                  test_gen_output, test_target)
        test_disc_loss = discriminator_loss(test_disc_real_output, test_disc_generated_output)

    # Registramos métricas
    with summary_writer.as_default():
        ss = step // 1000
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=ss)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=ss)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=ss)
        tf.summary.scalar('disc_loss', disc_loss, step=ss)

    return gen_total_loss, disc_loss, test_gen_total_loss, test_disc_loss

def fit(train_xy, test_xy, steps):
    # toma un lote, batch de pares (x,y)
    x, y = next(iter(test_xy.take(1)))
    start = time.time()

    # Emulamos un objeto history para visualizar las métricas del proceso
    # de entrenamiento
    history = {
        # Train set
        'train_gen_loss': np.zeros(steps),
        'train_disc_loss': np.zeros(steps),
        # Validation set
        'test_gen_loss': np.zeros(steps),
        'test_disc_loss': np.zeros(steps)
    }
    # Obtenemos un lote de imagenes de entrenamiento por cada paso realizado
    # Obtenemos simultáneamente un lote de imágenes de validación
    for (step, (x, y)), (xt, yt) in zip(train_xy.repeat().take(steps).enumerate(),
                                        test_xy.repeat().take(steps)):
        # Cada mil pasos previsualizamos el resultado del generador
        if ((step + 1) % 1000 == 0) and (step > 0):
            display.clear_output(wait=True)
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()
            generate_images(f"training_step{step}.png", generator, x, y)
            print(f"Step: {step // 1000}k")

        # Ejecutamos el paso de entrenamiento
        gen_loss, disc_loss, test_gen_loss, test_disc_loss = train_step(x, y, xt, yt, step)

        history['train_gen_loss'][step] = gen_loss
        history['train_disc_loss'][step] = disc_loss
        history['test_gen_loss'][step] = test_gen_loss
        history['test_disc_loss'][step] = test_disc_loss
        if (step + 1) % 10 == 0: print('.', end='', flush=True)

        # Cada cinco mil pasos guardamos un checkpoint
        if ((step + 1) % 5000 == 0) and (step > 0):
            checkpoint.save(file_prefix=checkpoint_prefix)

    return history

history = fit(train_xy, test_xy, steps=1000)

# Visualización de gráficas del entrenamiento
fig, ax = plt.subplots(2, 2, figsize=(10,6))
ax[0,0].plot(history['train_gen_loss'][:100])
ax[0,0].set_title('Generator train loss')
ax[0,1].plot(history['train_disc_loss'][:100])
ax[0,1].set_title('Discriminator train loss')
ax[1,0].plot(history['test_gen_loss'][:100])
ax[1,0].set_title('Generator val loss')
ax[1,1].plot(history['test_disc_loss'][:100])
ax[1,1].set_title('Discriminator val loss')
plt.savefig("figura9.png")

# Verificación del resultado del entrenamiento sobre el conjunto de datos de prueba
# Reestablecemos el último checkpoint generado
chkpnt = tf.train.latest_checkpoint(checkpoint_dir)
chkpnt = './training_checkpoints/ckpt-1'
checkpoint.restore(chkpnt)

step = 1
for x, y in test_xy.take(8):
    generate_images(f"testing_step{step}.png", generator, x, y)
    step += 1

# Dehaze de imágenes reales
real_dir = '/home/est_posgrado_manuel.suarez/data/ReDWeb-S/real'
xr_files  = glob(os.path.join(real_dir, 'RGB', '*.jpg'))
yr_files  = glob(os.path.join(real_dir, 'RGB', '*.jpg'))
xr_files.sort()
yr_files.sort()
xr_files=np.array(xr_files)
yr_files=np.array(yr_files)

real_x = tf.data.Dataset.list_files(xr_files, shuffle=False)
real_y = tf.data.Dataset.list_files(yr_files, shuffle=False)

real_xy = tf.data.Dataset.zip((real_x, real_y))
real_xy = real_xy.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
real_xy = real_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
real_xy = real_xy.batch(1)

# Verificación de carga de archivos por medio de la definición de los Dataset
for x, y in real_xy.take(1):
    display_images("real_images.png", x, y, rows=3)
    break

step = 1
for x, y in test_xy.take(5):
    generate_images(f"real_dehaze{step}.png", generator, x, y)
    step += 1
