from utils import *
from Model import *
from Genetic import *
import argparse
import os


import warnings
warnings.filterwarnings("ignore")

import logging
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Time series forecasting program')
parser.add_argument("--data_dir", help="folder of the dataset", default="datasets")
parser.add_argument("--models", help="Models used on the genetic algoritm can be RNN, TCN, NBEATS, TFT you can provide multiples with comas without any space", default="RNN,TCN,NBEATS,TFT")
parser.add_argument("--recursive", help="Prediction on models in recursive way can be True or False", default=True)
parser.add_argument("--tt_split", help="Percentage of the train set from 0 to 1",default="0.8")
parser.add_argument("--generations", help="amount of generations of the genetic algorithm",default="10")
parser.add_argument("--top", help="top models for cross on the genetic algorithm",default="5")
parser.add_argument("--cross", help="amount of crosses on the genetic algorithhm",default="4")
parser.add_argument("--init_pop", help="Initial population of the genetic algorithm",default="20")
parser.add_argument("--log_dir", help="Results of Model genetic Training",default="logs")
parser.add_argument("--mode", help="the mode of use of the script: Genetic for a genetic algoritm, Train for train a model with defined hyperparameters and Test for testing the model",default="Genetic")
parser.add_argument("--weights_dir", help="Weights of Model Training",default="weights")
parser.add_argument("--epoch",help="Epoch on Genetic or Train mode", default="50")
parser.add_argument("--pred_dir", help="dataframe of the predictions",default="predictions")
parser.add_argument("--n_pred", help="prediction horizon by default the amount of outputs",default="Output")

args = parser.parse_args()

data_dir = args.data_dir
models = args.models.split(',')
data_names = os.listdir(data_dir)
tt_split = float(args.tt_split)
recursive = bool(args.recursive)
gen = int(args.generations)
top = int(args.top)
crs = int(args.cross)
pop = int(args.init_pop)
log_dir = args.log_dir
wei_dir = args.weights_dir
Mode = args.mode
epoch = int(args.epoch)
pred_dir = args.pred_dir
if Mode == 'Genetic':
    Best_model =pd.DataFrame(columns=['Model','Dataset',"Fitness","Recursive"]) 
    for names in data_names:
        names = names.split('.')[0]
        train,test = data_preprocess(names,data_dir,tt_split)
        for model_name in models:
            Best = Genetic_algoritm (train, test, recursive = recursive, Generations = gen, top = top,
                                      cross = crs, init_pop = pop, model_name = model_name, epoch=epoch)
            df_g = make_dict(Best,model_name,names,recursive)
            df_g.to_csv(log_dir+"/Results_"+names+"_"+model_name+".csv", mode='a', index=False)
elif Mode == 'Train':
    for names in data_names:
        names = names.split('.')[0]
        train,test = data_preprocess(names,data_dir,tt_split)
        for model_name in models:
            best = pd.read_csv(log_dir+"/Results_"+names+"_"+model_name+".csv")
            param, recursive = dict_to_param(best)
            model = Model_selection(model_name, param, recursive, epoch)
            model.fit(train, verbose = True)
            model.save(wei_dir+"/Model_"+names+"_"+model_name+".pkl")
            forecast = model.historical_forecasts(test,start=test.get_timestamp_at_point(param[0]),forecast_horizon=param[1],retrain=False,verbose=False)
            print(r2_score(test.map(lambda x:x+1), forecast.map(lambda x:x+1)))
elif Mode == 'Test':
    for names in data_names:
        names = names.split('.')[0]
        train,test = data_preprocess(names,data_dir,tt_split)
        for model_name in models:
            best = pd.read_csv(log_dir+"/Results_"+names+"_"+model_name+".csv")
            param, recursive = dict_to_param(best)
            model = Model_selection(model_name, param, recursive, 0)
            model.fit(train, verbose = True)
            model.load_weights(wei_dir+"/Model_"+names+"_"+model_name+".pkl")
            if args.n_pred.isdigit():
                n_pred = int(args.n_pred)
            else:
                n_pred = param[1]
            forecast = model.predict(n_pred)
            predict = data_inversed(names, data_dir,tt_split,forecast)
            predict.to_csv(pred_dir+"/Predictions_"+names+"_"+model_name+".csv",index=False)
    