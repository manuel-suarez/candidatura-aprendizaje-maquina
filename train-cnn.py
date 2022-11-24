# Import
import os
import numpy as np
import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
from matplotlib import pyplot as plt

# TODO actualizar la ruta en referencia a los datos en el clúster
gdrive_dir = 'gdrive/MyDrive/DeepLearning'
local_dir = 'lgg-mri-segmentation-jpeg/kaggle_3m'

# Configuración de directorio
# TODO actualizar la expresión regular
ds_list = tf.data.Dataset.list_files(os.path.join(local_dir,'*/*[^mask].jpg'), shuffle=False)
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
  # TODO actualizar referencia a nombres de archivos de visualización
  image = load_file(os.path.join(local_dir,'TCGA_CS_4941_19960909','TCGA_CS_4941_19960909_11.jpg'))
  mask = load_file(os.path.join(local_dir,'TCGA_CS_4941_19960909','TCGA_CS_4941_19960909_11_mask.jpg'))
  print(type(image))
  print(type(mask))

  # Visualización
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
  ax[0].imshow(image)
  ax[0].set_title('Imagen')
  ax[1].imshow(mask[:,:,0])
  ax[1].set_title('Máscara')
  fig.tight_layout()
  plt.show()

test_load_file_and_visualize()

# Normalización de rango dinámico (0-1)
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.float32) / 255.0
  #input_mask -= 1
  return input_image, input_mask

# Función de carga de imágen y máscara
def load_image(file_path):
  # Descomponemos el nombre del archivo para obtener su nombre separado de su extensión
  file_components = tf.strings.split(file_path, '.')
  # Interpolamos el posfijo _mask al nombre del archivo para cargar la máscara
  mask_path = file_components[0] + '_mask.' + file_components[1]

  # Cargamos archivo y su máscara
  file_array = load_file(file_path)
  mask_array = load_file(mask_path)

  # Redimensionamos archivos para ajustarlo al tamaño de entrada de la red
  input_image = tf.image.resize(file_array, (128, 128))
  input_mask = tf.image.resize(mask_array, (128, 128))
  # Separamos canales
  input_image = input_image[:, :, :3]
  input_mask = input_mask[:, :, :1]

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
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
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
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# Visualización de imágenes
for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

# Construcción del modelo
# Usamos como modelo base MobileNetV2 especificando la forma de los datos de entrada y sin incluir la capa de clasificación
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

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
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

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
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Funciones auxiliares para visualización del modelo
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))