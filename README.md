# Time series using Genetic Algoritm
Este proyecto consisten en la generacion de un algoritmo que encuentra hyperparametros, entrena y testea el alcance de una base de datos con 4 modelos RNN, TCN, NBEATS y TFT.

## Creacion del ambiente
Se debe crear un ambiente e installar las librerias del archivo requirements.txt

## Ejecucion
Este algorimo cuenta con 3 modos:

### Genetic
Este modo se da cuando se ejecuta:

`python main.py --mode Genetic` o `python main.py`

Este algoritmo lo que hace es buscar las bases de datos dentro de la carpeta **datasets** (la cual puede ser cambiada con el comando `--data_dir`) las cuales deben tener el siguiente formato en el cual la columna Fecha es la primera y tiene que ser las estampas de tiempo, fechas u otro formato de tiempo, seguido de las columnas de los valores de la serie de tiempo por lo cual dependen del lugar, en caso de tener mas de un lugar se agregan en columnas aparte.

| Fecha | Lugar 1  | Lugar N |
| :------: |:--------:| :-----:|
| Timestamp 1 | Valor 1/Lugar 1 | Valor 1/Lugar N |
| Timestamp 2 | Valor 2/Lugar 1 | Valor 2/Lugar N |
| Timestamp N | Valor N/Lugar 1 | Valor N/Lugar N |

Posteriormente este algoritmo ejecuta un algoritmo genetico y los mejores resultados se pueden observar en la carpeta **logs** (la cual puede ser cambiada con el comando `--log_dir`)
si se desean cambiar los valores del algoritmo genetico puede usar `-h` para ver las opciones extras.

### Train
Este modo se da cuando se ejecuta:

`python main.py --mode Train`

Este algoritmo utiliza la base de datos dentron de la carpeta **datasets** y los resultados del algoritmo genetico de la carpeta **logs** y entrena el modelo guardandolo en la carpeta **weights** (la cual puede ser cambiada con el comando `--weights_dir`), es importante que si desea mejorar la metrica debe de aumentar las epocas con el comanddo `--epoch`.

### Test
Este modo se da cuando se ejecuta:

`python main.py --mode Test`

Este algoritmo utiliza la base de datos en la carpeta **datasets** y  los pesos en la carpeta **weights** para realizar una prediccion de los siguientes valores para cambiar la cantidad de valores a predecir utilice el comando 	`--n_pred`, y este generara un archivo csv con la prediccion y lo guardar en la carpeta **predictions** (la cual puede ser cambiada con el comando `--pred_dir`)


## Interfaz Grafica
Para ejecutar localmente la interfaz grafica usar el comando:
`streamlit run Streamlit.py`