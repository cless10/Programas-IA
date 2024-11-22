import tensorflow as tf
import tensorflow_datasets as tfds

#descargamos el dataset de imagenes, con esta linea ejecutar el archivo y se descargaran
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
#vemos la información de los metadatos (tiene 60000 datos para entrenamiento y 10000 para pruebas)
print(metadatos)

#ponemos los datos en variables diferentes para poder utilizarlos
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

#los metadatos ya contienen los nombres de las categorias que existen en el set, las asignamos a una variable para verlos
print('Esos son los nombres de las clases')
nombres_clases = metadatos.features['label'].names

print(nombres_clases)
print('\n')

#Normalizar los datos (Pasar de 0-255 a 0-1), hacemos que todas las entradas sean entre 0 y 1, 
#esto ayuda al entrenamiento
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255  #Aqui lo pasa de 0-255 a 0-1
    return imagenes, etiquetas

#Normalizar los datos de entrenamiento y pruebas con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#Agregar a cache (usar memoria en lugar de disco. así el entrenamiento es más rápido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#código para mostrar la primera imagen del dataset
for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28,28))    #Redimensionamos, cosas de tensores, se verá despues

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
#plt.show();                #muestra el grafico (pero se debe de tener instalado algun backends de GUI de matplolib) para solucionar el problema ver abajo
plt.savefig("prenda.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo


#código para mostrar las primeras 25 imagenes del dataset

plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28,28))    #Redimensionamos, cosas de tensores, se verá despues
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombres_clases[etiqueta])
#plt.show();                #muestra el grafico (pero se debe de tener instalado algun backends de GUI de matplolib) para solucionar el problema ver abajo
plt.savefig("prendas.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo


#Definimos el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),   #1 - blanco y negro (una capa), pone la matriz en una sola dimension, 784 neuronas donde se recibira cada pixel
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  #funcion de activación softmax, se usa como neurona de salida para redes de clasificacion
])

#Compilamos el modelo

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#Para que se haga más rápido, se realizará por lotes
TAMANO_LOTE = 32
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

print(num_ej_entrenamiento)
print(num_ej_pruebas)

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

#empezamos a entrenar la red
import math

#Entrenar
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))

#Resultado de la funcion de perdidad en cada epoca (cada vuelta)
plt.figure()
plt.title("Resultado de la funcion de perdida en cada epoca (cada vuelta)")
plt.xlabel("# Epoca (Vuelta)")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.savefig("perdidaPorEpoch.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo


#imprimimos 25 imagenes y se intentará predecir cada uno  (codigo de matlib)
import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i] 
    #plt.figure()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'  #Si le atino
    else:
        color = 'red'  #Ooops, no le atino

    plt.xlabel("{} {:2.0f}% ({})".format(
        nombres_clases[etiqueta_prediccion],
        100*np.max(arr_predicciones),
        nombres_clases[etiqueta_real],
        color=color
    ))

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real=arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0,1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')

filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))

for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)

plt.savefig("graficaResultado.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo


#Tomar cualquier indice del set de pruebas para ver su prediccion
imagen = imagenes_prueba[7]
imagenExp = imagen

imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print("")
print("Predicción: " + nombres_clases[np.argmax(prediccion[0])])

#Mostramos la imagen de la predicción
plt.figure()
plt.imshow(imagenExp, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
#plt.show();                #muestra el grafico (pero se debe de tener instalado algun backends de GUI de matplolib) para solucionar el problema ver abajo
plt.savefig("prenda_prediccion.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo

#Exportarlo a un navegador
modelo.save('modelo_exportado.h5')

#importamos tensorflowjs
#pip install tensorflowjs

#luego
'''
#Convertir el archivo h5 a formato de tensorflowjs
mkdir tfjs_target_dir
tensorflowjs_converter --input_format keras modelo_exportado.h5 tfjs_target_dir

#Veamos si si creo la carpeta
ls

#Veamos el contenido de la carpeta
ls tfjs_target_dir
'''