# A-Generalizable-Guide-to-Multi-Output-Spatio-temporal-Forecasting
Github repo for the paper "Building Better Forecasting Pipelines: A Generalizable Guide to Multi-Output Spatio-temporal Forecasting",This project consists in the generation of a genetic algorithm that finds hyperparameters, trains and tests the scope of a database with 4 models RNN, TCN, NBEATS and TFT.

## Virtual enviroment
You must create a virtual environment and install the libraries from the requirements.txt file.

## Execution
This algorithm has 3 modes:

### Genetic
This mode occurs when running:

`python main.py --mode Genetic` or `python main.py`

What this algorithm does is to look for the databases inside the **datasets** folder (which can be changed with the `--data_dir` command) which must have the following format in which the Date column is the first one and has to be the time stamps, dates or other time format, followed by the columns of the time series values so they depend on the location, in case of having more than one location they are added in separate columns.

| Date | Location 1  | Location N |
| :------: |:--------:| :-----:|
| Timestamp 1 | Value 1 - Location 1 | Value 1 - Location N |
| Timestamp 2 | Value 2 - Location 1 | Value 2 - Location N |
| Timestamp N | Value N - Location 1 | Value N - Location N |

Subsequently this algorithm runs a genetic algorithm and the best results can be observed in a **logs** folder (which can be changed with the `--log_dir` command).
if you want to change the values of the genetic algorithm you can use `-h` to see the extra options.

### Train
This mode occurs when running:

`python main.py --mode Train`

This algorithm uses the database inside the **datasets** folder and the results of the genetic algorithm from the **logs** folder and trains the model by saving it in a **weights** folder (which can be changed with the `--weights_dir` command), it is important that if you want to improve the metric you should increase the epochs with the `--epoch` command.

### Test
This mode occurs when running:

`python main.py --mode Test`

This algorithm uses the database in the **datasets** folder and the weights in the **weights** folder to make a prediction of the following values to change the amount of values to predict use the command `--n_pred`, and it will generate a csv file with the prediction and save it in a **predictions** folder (which can be changed with the command `--pred_dir`)


## Graphic Interface
There is a graphic interface that have all the Execution modes above, it can be access by executing the following command:
`streamlit run Streamlit.py`





