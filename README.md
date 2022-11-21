# Proyecto de Examen de Candidatura - Aprendizaje Máquina

## Proyecto del curso

1. Usando una base de datos RBG-D (ver https://arxiv.org/pdf/2201.05761.pdf) crear un conjunto de datos sintéticos de imagenes (limpias, con niebla) de acuerdo con el modelo (1) en https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5567108.
2. Luego una vez construida la base de datos. Implementar un modelo pix2pix para remover niebla en imágenes. 
3. Comparar vs la implementación simple UNet (sin entrenamiento GAN). Usar métricas MSE y MAE.
4. Demostrar con ejemplos con datos reales.
5. Discutir resultados, dificultad de implementacion, estabilidad del entrenamiento. 

## Scripts

- **generate-haze-images.py**: agrega el elemento de niebla al conjunto de imágenes de acuerdo con el modelo en https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5567108
- **train-gan.py**: entrenamiento con un modelo pix2pix para dehazing
