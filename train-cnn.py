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