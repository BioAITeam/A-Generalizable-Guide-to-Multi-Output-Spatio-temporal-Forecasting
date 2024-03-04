import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def dict_to_param(dataframe):
  data = dataframe.iloc[-1].values.flatten().tolist()
  param = data[1:-3]
  param =  [float(x) for x in param]
  param =  [int(x) if x >= 1 else float(x) for x in param]
  recursive = data[-1]
  return param,recursive


def make_dict(Best,model_name,data_name,recursive):
  if model_name == "RNN":
    keys = ["Generation","Input","Output","Unit","Hidden","Layers","Dropout","Fitness","Model","Dataset","Recursive"]
  elif model_name == "TCN":
    keys = ["Generation","Input","Output","Dropout","Dilations","Kernel","Filters","Fitness","Model","Dataset","Recursive"]
  elif model_name == "NBEATS":
    keys = ["Generation","Input","Output","Blocks","Layers","Width","Batch","Fitness","Model","Dataset","Recursive"]
  elif model_name == "TFT":
    keys = ["Generation","Input","Output","Hidden","LSTM","Attention","Dropout","Batch","Fitness","Model","Dataset","Recursive"]
  else:
    print("Model name incorrect")
    model = []
  return pd.DataFrame(np.column_stack([Best, [model_name for x in range(Best.shape[0])], [data_name for x in range(Best.shape[0])],[recursive for x in range(Best.shape[0])]]), columns=keys)

def prepare_data(data, input_len, output_len):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + output_len])
    return np.array(X), np.array(y)

def data_preprocess(data_name, data_dir,TT_split):
  df = pd.read_csv(data_dir+'/'+data_name+'.csv')
  df=df.drop(columns=['Unnamed: 0'])
  data = TimeSeries.from_dataframe(df,'Fecha',df.columns[1:])
  train_idx = round(len(data)*TT_split)
  data_train = data[:train_idx]
  data_test = data[train_idx:]
  scaler_dataset = Scaler()
  # scaler is fit on training set only to avoid leakage
  train_scaled = scaler_dataset.fit_transform(data_train)
  test_scaled = scaler_dataset.transform(data_test)
  return train_scaled,test_scaled

def data_inversed(data_name, data_dir,TT_split,forecast):
  df = pd.read_csv(data_dir+'/'+data_name+'.csv')
  df=df.drop(columns=['Unnamed: 0'])
  data = TimeSeries.from_dataframe(df,'Fecha',df.columns[1:])
  train_idx = round(len(data)*TT_split)
  data_train = data[:train_idx]
  data_test = data[train_idx:]
  scaler_dataset = Scaler()
  # scaler is fit on training set only to avoid leakage
  train_scaled = scaler_dataset.fit_transform(data_train)
  predict = scaler_dataset.inverse_transform(forecast)
  return predict