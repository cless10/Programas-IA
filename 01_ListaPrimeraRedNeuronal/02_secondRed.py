#mismo ejemplo que el anterior, pero aplicado a una ecuación simple x = y * 2
import tensorflow as tf;
import numpy as np;

equis = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype=float)             #que vamos a predecir
ye = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=float)                  #resultados para entrenamiento

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

#se agregan dos capas intermedias con tres neuronas
#oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
#oculta2 = tf.keras.layers.Dense(units=3)
#salida = tf.keras.layers.Dense(units=1)
#modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(equis, ye, epochs=400, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt;

plt.title("Grafico");
plt.xlabel("# Epoca");
plt.ylabel("Magnitud de perdida");
plt.plot(historial.history["loss"]);
#plt.show();                #muestra el grafico (pero se debe de tener instalado algun backends de GUI de matplolib) para solucionar el problema ver abajo
plt.savefig("mygraph.png")  #Para no instalar un backend, mejor exportamos el grafico a un archivo

valor_predecir = -20;
print("");
print("Hagamos una predicción!");
print("Predicción para el valor de y, cuando x = " + str(valor_predecir));
resultado = modelo.predict(x=np.array([valor_predecir]));
print("El resultado, el valor de y es: " + str(resultado) + "!");


#pedimos al usuario un valor de x para que la red prediga el valor de y
print("Por favor Introduzca un valor de x para realizar la predicción del valor y: ")
varX = float(input());
print("");
print("Hagamos La predicción!");
print("Predicción para el valor de y, cuando x = " + str(varX));
resultado = modelo.predict(x=np.array([varX]));
print("El resultado, el valor de y es: " + str(resultado) + "!");



print("");
print("Variables internas del modelo");
print(capa.get_weights());
print("");
#print(oculta1.get_weights());
#print(oculta2.get_weights());
#print(salida.get_weights());



"""   (comentario multilinea)
https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so

Solución 1: es instalar el backend GUItk
Encontré una solución a mi problema (gracias a la ayuda de ImportanceOfBeingErnest ).

Todo lo que tenía que hacer era instalar tkintera través de la terminal bash de Linux usando el siguiente comando:

sudo apt-get install python3-tk
en lugar de instalarlo con pipo directamente en el entorno virtual en Pycharm.

Solución 2: instale cualquiera de los matplotlibbackends de GUI admitidos
la solución 1 funciona bien porque obtienes un backend GUI... en este caso, elTkAgg
sin embargo, también puede solucionar el problema instalando cualquiera de los backends de GUI de matplolib como Qt5Agg, GTKAgg, Qt4Agg, etc.
por ejemplo, pip install pyqt5también solucionará el problema
NOTA:

por lo general, este error aparece cuando pip instala matplotlib y está tratando de mostrar un gráfico en una ventana de GUI y no tiene un módulo de python para la visualización de GUI.
Los autores de matplotlibhicieron que los departamentos de software de pypi no dependan de ningún backend de GUI porque algunas personas necesitan matplotlib sin ningún backend de GUI.
"""