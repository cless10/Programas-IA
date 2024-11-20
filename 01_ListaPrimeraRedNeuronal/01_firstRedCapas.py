#Mi primera red neuronal con Python y tensorflow
import tensorflow as tf;
import numpy as np;

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#capa2 = tf.keras.layers.Dense(units=5, input_shape=[5])        #se pueden agregar mas capas, con n neuronas
#capa3 = tf.keras.layers.Dense(units=1, input_shape=[1])
##modelo = tf.keras.Sequential([capa])
#modelo = tf.keras.Sequential([capa, capa2, capa3])

#se agregan dos capas intermedias con tres neuronas
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Por favor Introduzca un valor flotante de grados centígrados para realizar la predicción: ")
aPredecir = float(input());
print("Por favor Introduzca un valor entero para realizar el entrenamiento (iteraciones, se recomienda al menos 1000): ")
iteraciones = int(input());

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=iteraciones, verbose=False)
print("Modelo entrenado!")

print("Solo con 7 valores para iniciar el aprendizaje y " + str(iteraciones) + " epoch (iteraciones de entrenamiento)!");

import matplotlib.pyplot as plt;

plt.xlabel("# Epoca");
plt.ylabel("Magnitud de perdida");
plt.plot(historial.history["loss"]);
plt.savefig("mygraph_firstNet_Capas.png");

print("");
print("Hagamos una predicción!");
#aPredecir = 37;
resultado = modelo.predict(x=np.array([aPredecir]));
print("Predecir " + str(aPredecir) + " celsius:");
print("El resultado es: " + str(resultado) + " fahrenheit!");
print("\nEste programa genera una gráfica sobre el modelo. Puedes revisar el archivo 'mygraph_firstNet.png'\n");

print("");
print("Variables internas del modelo capa oculta 1");
print(oculta1.get_weights());
print("Variables internas del modelo capa oculta 2");
print(oculta2.get_weights());
print("Variables internas del modelo capa salida");
print(salida.get_weights());
#print(oculta1.get_weights());
#print(oculta2.get_weights());
#print(salida.get_weights());


print("\nInformación adicional:");
print("¿Qué es Mean_squared_error en Python? \n"
"El error cuadrático medio es el promedio del cuadrado de la diferencia entre  \n"
"los valores observados y predichos de una variable . En Python, el MSE se  \n"
"puede calcular con bastante facilidad, especialmente con el uso de listas. \n\n"
"Un Epoch o época es cuando todos los datos de entrenamiento se usan a la vez  \n"
"y se define como el número total de iteraciones de todos los datos de  \n"
"entrenamiento en un ciclo para entrenar el modelo de aprendizaje automático.\n");

'''
¿Qué es Mean_squared_error en Python?
El error cuadrático medio es el promedio del cuadrado de la diferencia entre 
los valores observados y predichos de una variable . En Python, el MSE se 
puede calcular con bastante facilidad, especialmente con el uso de listas.

Un Epoch o época es cuando todos los datos de entrenamiento se usan a la vez 
y se define como el número total de iteraciones de todos los datos de 
entrenamiento en un ciclo para entrenar el modelo de aprendizaje automático.
'''