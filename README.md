# A-Generalizable-Guide-to-Multi-Output-Spatio-temporal-Forecasting
Github repo for the paper "Building Better Forecasting Pipelines: A Generalizable Guide to Multi-Output Spatio-temporal Forecasting",This project consists in the generation of a genetic algorithm that finds hyperparameters, trains and tests the scope of a database with 4 models RNN, TCN, NBEATS and TFT.

## Virtual enviroment
You must create a virtual environment and install the libraries from the requirements.txt file.

## Execution
This algorithm has 3 modes:

### Genetic
This mode occurs when running:

`python main.py --mode Genetic` o `python main.py`

What this algorithm does is to look for the databases inside the **datasets** folder (which can be changed with the `--data_dir` command) which must have the following format in which the Date column is the first one and has to be the time stamps, dates or other time format, followed by the columns of the time series values so they depend on the location, in case of having more than one location they are added in separate columns.

| Date | Location 1  | Location N |
| :------: |:--------:| :-----:|
| Timestamp 1 | Value 1 - Location 1 | Value 1 - Location N |
| Timestamp 2 | Value 2 - Location 1 | Value 2 - Location N |
| Timestamp N | Value N - Location 1 | Value N - Location N |

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
=======



