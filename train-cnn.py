# Import
import os
import numpy as np
import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
from matplotlib import pyplot as plt

# Ruta de los datos en el clúster
local_dir = '/home/est_posgrado_manuel.suarez/data/ReDWeb-S/trainset'

# Configuración de directorio
ds_list = tf.data.Dataset.list_files(os.path.join(local_dir,'haze','beta05','*.jpg'), shuffle=False)
val_size = int(len(ds_list) * 0.2)
ds_train = ds_list.skip(val_size)
ds_test  = ds_list.take(val_size)

# Carga de archivo
def load_file(filepath):
  # Abrimos archivo
  image_file = tf.io.read_file(filepath)
  # Decodificamos imagen
  return tf.image.decode_jpeg(image_file)

# Función para validar funcionalidad de la función cargando y desplegando un archivo y máscara correspondiente
def test_load_file_and_visualize():
  image = load_file(os.path.join(local_dir,'haze','beta05','4757359274_6fab3f7680_b.jpg'))
  mask = load_file(os.path.join(local_dir,'RGB','4757359274_6fab3f7680_b.jpg'))
  print(type(image))
  print(type(mask))

  # Visualización
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
  ax[0].imshow(image)
  ax[0].set_title('Imagen')
  ax[1].imshow(mask)
  ax[1].set_title('Máscara')
  fig.tight_layout()
  plt.savefig("figura1.png")
  plt.close()

test_load_file_and_visualize()

# Normalización de rango dinámico (0-1)
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.float32) / 255.0
  #input_mask -= 1
  return input_image, input_mask

# Función de carga de imágen y máscara
def load_image(file_path):
  # Descomponemos el nombre del archivo para obtener su separación en directorios
  # Datos con niebla: local_dir = '/home/est_posgrado_manuel.suarez/data/ReDWeb-S/trainset/haze/beta25'
  # Datos clean:      local_dir = '/home/est_posgrado_manuel.suarez/data/ReDWeb-S/trainset/RGB'
  # Posiciones:                   0   1               2               3      4        5
  file_components = tf.strings.split(file_path, '/')
  # Interpolamos el posfijo _mask al nombre del archivo para cargar la máscara
  mask_path = file_components[0] + '/' + \
              file_components[1] + '/' + \
              file_components[2] + '/' + \
              file_components[3] + '/' + \
              file_components[4] + '/' + \
              file_components[5] + '/' + \
              'RGB'              + '/' + \
              file_components[8]

  # Cargamos archivo y su máscara
  file_array = load_file(file_path)
  mask_array = load_file(mask_path)

  # Redimensionamos archivos para ajustarlo al tamaño de entrada de la red
  input_image = tf.image.resize(file_array, (256, 256))
  input_mask = tf.image.resize(mask_array, (256, 256))
  # Devolvemos las imágenes RGB (notar que ésto provoca el error con la función de costo SparseCategoricalCrossentropy
  # ya que no se estarán comparando etiquetas por bit sino la imagen RGB completa por lo que se debe modificar dicha
  # función para usar MAE o MSE
  # input_image = input_image[:, :, :3]
  # input_mask = input_mask[:, :, :1]

  # Normalizamos datos
  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

# Parámetros del modelo
TRAIN_LENGTH = len(ds_train)#info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Definición de dataset con aplicación de la imagen de carga de imágenes al dataset de nombres de archivos
img_train = ds_train.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
img_test = ds_test.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Aumentación
class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    if tf.random.uniform(()) > 0.5:
      inputs = tf.image.flip_left_right(inputs)
      labels = tf.image.flip_left_right(labels)
    return inputs, labels

# Definición de lotes del conjuno de entrenamiento y prueba
train_batches = (
    img_train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = img_test.batch(BATCH_SIZE)

# Función auxiliar de despliegue de imagen
def display(figname, display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig(figname)
  plt.close()

# Visualización de imágenes
for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display("figura2.png", [sample_image, sample_mask])

# Construcción del modelo
# Usamos como modelo base MobileNetV2 especificando la forma de los datos de entrada y sin incluir la capa de clasificación
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

# Usamos las capas de activación
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
# Definimos salidas del modelo
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Creamos modelo base para extracción de características
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

# En el primer ciclo de entrenamiento congelamos los pesos del modelo base
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  # Capa de entrada
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Capas de downsampling y upsampling

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # Capa convolucional de salida (para 2a fase de entrenamiento)
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# Construcción y compilación del modelo
model = unet_model(output_channels=3)
model.compile(optimizer='adam',
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # Modificamos la función de costo por MAE o MSE ya que las entradas y salidas tienen el mismo número
              # de canales ya que no estamos haciendo segmentación
              #loss=tf.keras.losses.MeanSquaredError(reduction="auto"),
              loss=tf.keras.losses.MeanAbsoluteError(reduction="auto"),
              metrics=['accuracy'])

# Funciones auxiliares para visualización del modelo
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(figname, dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      prediction = model.predict(image)
      display(figname, [image[0], mask[0], prediction[0]])
  else:
    prediction = model.predict(sample_image[tf.newaxis, ...])
    display(figname, [sample_image, sample_mask, prediction[0]])


# Proceso de entrenamiento en tres pasos: 1.-Capas de salida - 2.-Capa convolucional - 3.-Todas las capas
EPOCHS = 30
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(ds_test)//BATCH_SIZE//VAL_SUBSPLITS

# Primera etapa
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(f"training1_{epoch}_result.png")
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("training1_results.png")
plt.close()

show_predictions("predictions1_result.png", test_batches, 3)

# Segunda etapa
model.layers[-1].trainable = True
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(f"training2_{epoch}_result.png")
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("training2_results.png")
plt.close()

show_predictions("predictions2_result.png", test_batches, 3)

# Tercera etapa
down_stack.trainable = True
for layer in model.layers:
  layer.trainable = True
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(f"training3_{epoch}_result.png")
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("training3_results.png")
plt.close()

show_predictions("predictions3_result.png", test_batches, 3)
