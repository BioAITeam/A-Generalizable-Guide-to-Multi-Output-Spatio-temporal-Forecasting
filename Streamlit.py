import streamlit as st
import pandas as pd
from utils import *
from Model import *
from Genetic_Streamlit import *
import os
import tkinter as tk
from tkinter import filedialog


def intro():

    st.write("# Welcome to the 'Nombre del producto'📊")

    st.sidebar.success("Select an option")

    st.markdown(
        """
        'nombre del producto'is a system for generating time series forecasts.
        for that we have multiple options.
        **👈 Select an option** to see the available ones
        ### Database
        Before you start using the system you should organize the information in your database in the following way.
        in the following way, in which we have in the columns the places and in the rows the times.
    
        


    """
    )
    @st.cache_data
    def get_UN_data():
        demo_df_url = "datasets\Data_Predict_Temp_Stations.csv"
        df = pd.read_csv(demo_df_url)
        df=df.drop(columns=['Unnamed: 0'])
        return df


    df = get_UN_data()
    countries = st.multiselect(
        "Choose the locations", list(df.columns), list(list(df.columns[:4]))
    )
    if not countries:
        st.error("Please select at least one location")
    else:
        data = df.loc[:10,countries]
        st.write("### Sample Database", data.sort_index())



    st.markdown(
        """
        ## Modes of use:

        ### Genetic
        Uses a genetic algorithm to train one to more models on the best result.
        
        ### Training
        trains the best result obtained from the genetic algorithm and saves the model for later use.
        further use.

        ### Testing
        Allows to test the saved models with the information from the database or new ones.
        
        """
        )

def genetic():
    st.write("# Selection of the best model :dna:")

    st.markdown(
        """
        This module is to be able to select the best parameters for the model, for this you must select some types of information as shown below. 
        select some types of information as shown below.

        ## Databases and results folders

        Here you select the folders where the database files are and where you want to save the results.

        """
        )
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)


    col1, col2 = st.columns(2)

    if "button1" not in st.session_state:
        st.session_state["button1"] = False

    if "button2" not in st.session_state:
        st.session_state["button2"] = False
    
    if "button3" not in st.session_state:
        st.session_state["button3"] = False

    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = " "
    
    if "log_dir" not in st.session_state:
        st.session_state["log_dir"] = " "


    
    with col1:
        if st.button("Database"):
            st.session_state["button1"] = True
        
        if st.session_state["button1"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button1"] = False
            
        st.write(st.session_state["data_dir"])

    with col2:
        if st.button("Results"):
            st.session_state["button2"] = True

        if st.session_state["button2"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button2"] = False
            
        st.write(st.session_state["log_dir"])

    st.markdown("""
                ## Training Parameters
                Here you modify the parameters for the model training in the Model column and for the Genetic algorithm in the Genetic column.
                """
                )

    col3, col4 = st.columns(2)

    with col3:
        st.write('### Model')
        tt_split = st.slider('Training dataset size', 0.0, 1.0, 0.7)
        models = st.multiselect("Choose the models", ["RNN","TCN","NBEATS","TFT"], ["RNN"])
        recursive = st.toggle('Recursive Forecasting')
        epoch = st.number_input('Number of epochs',value=50)
        output_toggle = st.toggle('Fixed Output')
        if output_toggle:
            output_value = st.number_input('Output value',value=10)
        else:
            output_value = False


    with col4:
        st.write('### Genetic')
        gen = st.number_input('Number of generations',value=10)
        top = st.number_input('Number of models to crossover',value=5)
        crs = st.number_input('Number of crosses',value=4)
        pop = st.number_input('Initial model population',value=10)


    
    poss_err = 8
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select database folder")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the results folder")

    if models:
        poss_err -=1
    else:
        st.error("Please select at least one model")
    
    if epoch > 0:
        poss_err -=1
    else:
        st.error("The minimum number of epochs is 1.")

    if gen > 0:
        poss_err -=1
    else:
        st.error("The minimum number of generations is 1.")

    if pop > 1:
        poss_err -=1
    else:
        st.error("The minimum number of population is 2.")

    if top < pop:
        poss_err -=1
    else:
        st.error("The maximum number of models to cross is the population size.")

    if crs >0 :
        poss_err -=1
    else:
        st.error("The minimum number of crosses is 1.")


    st.markdown("""
                ## Search of the best parameters
                Once you have solved all the errors to run the genetic algorithm, click on the button
                click on the button.
                """
                )

    if st.button("Run genetic algorithm"):
            st.session_state["button3"] = True
    if poss_err == 0 and  st.session_state["button3"]:
        data_names = os.listdir(st.session_state["data_dir"])
        Best_model =pd.DataFrame(columns=['Model','Dataset',"Fitness","Recursive"]) 
        for names in data_names:
            st.write("#### Database:", data_names[0])
            names = names.split('.')[0]
            train,test = data_preprocess(names,st.session_state["data_dir"],tt_split)
        for model_name in models:
                st.write("#### Processed Model:", model_name)
                Best = Genetic_algoritm (train, test, recursive = recursive, Generations = gen, top = top,
                                    cross = crs, init_pop = pop, model_name = model_name, epoch=epoch, output=output_value)
                df_g = make_dict(Best,model_name,names,recursive)
                df_g.to_csv(st.session_state["log_dir"] +"/Results_"+names+"_"+model_name+".csv", mode='a', index=False)
        st.session_state["button3"] = False
    else:
        st.error("There are pending errors to be solved")
 

def train():
    st.write("# Training of the best parameters :mechanical_arm:")

    st.markdown(
        """
        This module is to be able to train the model with the best parameters which are obtained in the Genetic tab.

        ## Databases, results and weights folder

        Here select the folders where the database files are, the results folder where the results of the genetics tab
        are saved and the weights folder where the trained model is saved.
        the results of the genetics tab and the weights folder where we saved the trained model.

        """
        )

    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)


    col1, col2, col3 = st.columns(3)


    if "button4" not in st.session_state:
        st.session_state["button4"] = False

    if "button5" not in st.session_state:
        st.session_state["button5"] = False

    if "button6" not in st.session_state:
        st.session_state["button6"] = False
    
    if "button7" not in st.session_state:
        st.session_state["button7"] = False

    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = " "
    
    if "log_dir" not in st.session_state:
        st.session_state["log_dir"] = " "

    if "wei_dir" not in st.session_state:
        st.session_state["wei_dir"] = " "

    with col1:
        if st.button("Database"):
                st.session_state["button4"] = True
            
        if st.session_state["button4"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button4"] = False
            
        st.write(st.session_state["data_dir"])
    
    with col2:
        if st.button("Results"):
            st.session_state["button5"] = True

        if st.session_state["button5"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button5"] = False
            
        st.write(st.session_state["log_dir"])
    
    with col3:
        if st.button("Weights"):
            st.session_state["button6"] = True

        if st.session_state["button6"]:
            st.session_state["wei_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button6"] = False
            
        st.write(st.session_state["wei_dir"])

    st.markdown("""
                ## Training Parameters
                Here you modify the parameters for training the model.
                """
                )
    tt_split = st.slider('Training dataset size', 0.0, 1.0, 0.7)

    models = st.multiselect("Choose the models", ["RNN","TCN","NBEATS","TFT"], ["RNN"])

    epoch = st.number_input('Number of epochs',value=100)

    poss_err = 5
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select database folder")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the folder of the results") 

    if not st.session_state["wei_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the weights folder") 

    if models:
        poss_err -=1
    else:
        st.error("Please select at least one model") 
    
    if epoch > 0:
        poss_err -=1
    else:
        st.error("The minimum number of epochs is 1.")


    st.markdown("""
                ## Training
                Once you have solved all the errors to run the genetic algorithm, click on the
                click on the button.
                """
                )

    if st.button("Start Training"):
            st.session_state["button7"] = True
    if poss_err == 0 and  st.session_state["button7"]:
        data_names = os.listdir(st.session_state["data_dir"])
        for names in data_names:
            names = names.split('.')[0]
            st.write("#### Database:", data_names[0])
            train,test = data_preprocess(names,st.session_state["data_dir"],tt_split)
            for model_name in models:
                best = pd.read_csv(st.session_state["log_dir"]+"/Results_"+names+"_"+model_name+".csv")
                param, recursive = dict_to_param(best)
                model = Model_selection(model_name, param, recursive, epoch)
                my_bar = st.progress(0, text='Training:Epoch 0')
                for e in range(epoch):
                    model.fit(train, verbose = True,epochs=1)
                    progress = 1*(e+1)/epoch
                    my_bar.progress(progress, text='Training:Epoch '+str(e+1))
                model.save(st.session_state["wei_dir"]+"/Model_"+names+"_"+model_name+".pkl")
                forecast = model.historical_forecasts(test,start=test.get_timestamp_at_point(param[0]),forecast_horizon=param[1],retrain=False,verbose=False)
                st.write(r2_score(test, forecast))
        st.session_state["button7"] = False
    else:
        st.error("There are pending errors to be solved")

def test():
    st.write("# Predicting with the best trained model 	:test_tube:")

    st.markdown(
        """
        This module is for testing the model with the best training parameters, which is obtained in the Training tab.

        ## Databases, results and weights folder

        Here you select the folders where the database files are stored, the results folder where the results from the genetics tab 
        the results of the genetics tab, the weights folder where the results of the training tab are saved and the pred where the predictions are saved.
        the pred where the predictions will be saved.

        """
        )
    
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)


    col1, col2, col3, col4 = st.columns(4)


    if "button8" not in st.session_state:
        st.session_state["button8"] = False

    if "button9" not in st.session_state:
        st.session_state["button9"] = False

    if "button10" not in st.session_state:
        st.session_state["button10"] = False
    
    if "button11" not in st.session_state:
        st.session_state["button11"] = False
    
    if "button12" not in st.session_state:
        st.session_state["button12"] = False

    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = " "
    
    if "log_dir" not in st.session_state:
        st.session_state["log_dir"] = " "

    if "wei_dir" not in st.session_state:
        st.session_state["wei_dir"] = " "

    if "pred_dir" not in st.session_state:
        st.session_state["pred_dir"] = " "
    
    with col1:
        if st.button("Database"):
                st.session_state["button8"] = True
            
        if st.session_state["button8"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button8"] = False
            
        st.write(st.session_state["data_dir"])
    
    with col2:
        if st.button("Results"):
            st.session_state["button9"] = True

        if st.session_state["button9"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button9"] = False
            
        st.write(st.session_state["log_dir"])
    
    with col3:
        if st.button("Weights"):
            st.session_state["button10"] = True

        if st.session_state["button10"]:
            st.session_state["wei_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button10"] = False
            
        st.write(st.session_state["wei_dir"])

    with col4:
        if st.button("Predictions"):
            st.session_state["button12"] = True

        if st.session_state["button12"]:
            st.session_state["pred_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button12"] = False
            
        st.write(st.session_state["pred_dir"])
    st.markdown("""
                ## Parameters for forecasting
                Here you modify the parameters to predict future values.
                """
                )
    models = st.multiselect("Choose the models", ["RNN","TCN","NBEATS","TFT"], ["RNN"])
    tt_split = st.slider('Training dataset size', 0.0, 1.0, 0.7)
    n_pred = st.number_input('Number of values to predict',value=2)

    if st.button("Predict"):
            st.session_state["button11"] = True

    poss_err = 6
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select database folder")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the folder of the results") 

    if not st.session_state["wei_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the weights folder") 

    if models:
        poss_err -=1
    else:
        st.error("Please select at least one model") 

    if not st.session_state["pred_dir"] == " ":
        poss_err -=1
    else:
        st.error("Please select the prediction folder")
    
    if n_pred > 0:
        poss_err -=1
    else:
        st.error("The minimum number of predictions is 1.")

    if poss_err == 0 and  st.session_state["button11"]:
        data_names = os.listdir(st.session_state["data_dir"])
        for names in data_names:
            names = names.split('.')[0]
            train,test = data_preprocess(names,st.session_state["data_dir"],tt_split)
            for model_name in models:
                best = pd.read_csv(st.session_state["log_dir"]+"/Results_"+names+"_"+model_name+".csv")
                param, recursive = dict_to_param(best)
                model = Model_selection(model_name, param, recursive, 0)
                model.fit(train, verbose = True)
                model.load_weights(st.session_state["wei_dir"]+"/Model_"+names+"_"+model_name+".pkl")
                forecast = model.predict(n_pred)
                predict = data_inversed(names, st.session_state["data_dir"],tt_split,forecast)
                predict.to_csv(st.session_state["pred_dir"]+"/Predictions_"+names+"_"+model_name+".csv",index=False)

    

page_names_to_funcs = {
    "Home": intro,
    "Genetic": genetic,
    "Training":train,
    "Testing":test,
}
st.set_page_config(page_title="Nombre del codigo", page_icon="📈")
demo_name = st.sidebar.selectbox("Select a function", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()