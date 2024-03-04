import streamlit as st
import pandas as pd
from utils import *
from Model import *
from Genetic_Streamlit import *
import os
import tkinter as tk
from tkinter import filedialog


def intro():

    st.write("# Bienvenido al 'Nombre del producto'📊")

    st.sidebar.success("Seleccione una opcion")

    st.markdown(
        """
        'nombre del producto'es un sistema para generar pronosticos de series de tiempo
        para eso tenemos muliples opciones.
        **👈 Seleccione una opcion** para ver las disponibles
        ### Base de datos
        Antes de empezar a usar el sistema debe organizar la informacion de su base de datos
        de la siguiente manera, en la cual tenemos en las columnas los lugares y en las filas los tiempos.
    
        


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
        "Elige los lugares", list(df.columns), list(list(df.columns[:4]))
    )
    if not countries:
        st.error("Por favor seleccione al menos un lugar")
    else:
        data = df.loc[:10,countries]
        st.write("### Base de datos de Muestra", data.sort_index())



    st.markdown(
        """
        ## Modos de Uso:

        ### Genetico
        Utiliza un algoritmo genetico para entrenar uno a mas modelos en el mejor resultado.
        
        ### Entrenamiento
        entrena el mejor resultado obtenido del algoritmo genetico y guarda el modelo para 
        su posterior uso.

        ### Testeo
        Permite probar los modelos guardados con la informacion de la base de datos o nueva.
        
        """
        )

def genetic():
    st.write("# Seleccion del mejor modelo:dna:")

    st.markdown(
        """
        Este Modulo es para poder seleccionar los mejores parametros para el modelo, para ello se debe 
        seleccionar algunos tipos de informacion como la que aparece a continucion.

        ## Carpeta de las Bases de datos y resultados

        Aqui seleccione las carpetas donde estan los archivos de bases de datos y donde desea guardar los resultados.

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
        if st.button("Base de datos"):
            st.session_state["button1"] = True
        
        if st.session_state["button1"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button1"] = False
            
        st.write(st.session_state["data_dir"])

    with col2:
        if st.button("Resultados"):
            st.session_state["button2"] = True

        if st.session_state["button2"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button2"] = False
            
        st.write(st.session_state["log_dir"])

    st.markdown("""
                ## Parametros para Entrenar
                Aqui se modifican los parametros para el entrenamiento del modelo en la columna Modelo
                y para el algoritmo Genetico en la columna Genetico.
                """
                )

    col3, col4 = st.columns(2)

    with col3:
        st.write('### Modelo')
        tt_split = st.slider('Tamaño del dataset de entrenamiento', 0.0, 1.0, 0.7)
        models = st.multiselect("Elige los modelos", ["RNN","TCN","NBEATS","TFT"], ["RNN"])
        recursive = st.toggle('Predición Recursiva')
        epoch = st.number_input('Numero de epocas',value=50)
        


    with col4:
        st.write('### Genetico')
        gen = st.number_input('Numero de generaciones',value=10)
        top = st.number_input('Numero de modelos para cruzar',value=5)
        crs = st.number_input('Numero de cruces',value=4)
        pop = st.number_input('Poblacion inicial de modelos',value=10)


    
    poss_err = 8
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de la base de datos")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de los resultados") 

    if models:
        poss_err -=1
    else:
        st.error("Por favor seleccione al menos un modelo") 
    
    if epoch > 0:
        poss_err -=1
    else:
        st.error("El numero minimo de epocas es de 1")

    if gen > 0:
        poss_err -=1
    else:
        st.error("El numero minimo de generaciones es de 1")

    if pop > 1:
        poss_err -=1
    else:
        st.error("El numero minimo de poblacion es de 2")

    if top < pop:
        poss_err -=1
    else:
        st.error("El numero maximo de modelos a cruzar es el tamaño de la poblacion")

    if crs >0 :
        poss_err -=1
    else:
        st.error("El numero minimo de cruces es 1")


    st.markdown("""
                ## Busqueda de los mejores parametros
                Una vez solucionado todos los errores para ejecutar el algoritmo genetico
                de al boton.
                """
                )

    if st.button("Correr algortimo genetico"):
            st.session_state["button3"] = True
    if poss_err == 0 and  st.session_state["button3"]:
        data_names = os.listdir(st.session_state["data_dir"])
        Best_model =pd.DataFrame(columns=['Model','Dataset',"Fitness","Recursive"]) 
        for names in data_names:
            st.write("#### Base de dato:", data_names[0])
            names = names.split('.')[0]
            train,test = data_preprocess(names,st.session_state["data_dir"],tt_split)
        for model_name in models:
                st.write("#### Modelo Procesado:", model_name)
                Best = Genetic_algoritm (train, test, recursive = recursive, Generations = gen, top = top,
                                    cross = crs, init_pop = pop, model_name = model_name, epoch=epoch)
                df_g = make_dict(Best,model_name,names,recursive)
                df_g.to_csv(st.session_state["log_dir"] +"/Results_"+names+"_"+model_name+".csv", mode='a', index=False)
        st.session_state["button3"] = False
    else:
        st.error("hay errores pendientes por solucionar")
 

def train():
    st.write("# Entrenamiento de los mejores parametros :mechanical_arm:")

    st.markdown(
        """
        Este Modulo es para poder entrenar el modelo con los mejores parametros los cuales se obtienen en la pestaña de Genetico.

        ## Carpeta de las Bases de datos, resultados y pesos

        Aqui seleccione las carpetas donde estan los archivos de bases de datos, la carpeta de resultados donde se guardaron 
        los resultados de la pestaña de genetico y la carpeta de pesos donde guardamos el modelo entrenado.

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
        if st.button("Base de datos"):
                st.session_state["button4"] = True
            
        if st.session_state["button4"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button4"] = False
            
        st.write(st.session_state["data_dir"])
    
    with col2:
        if st.button("Resultados"):
            st.session_state["button5"] = True

        if st.session_state["button5"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button5"] = False
            
        st.write(st.session_state["log_dir"])
    
    with col3:
        if st.button("Pesos"):
            st.session_state["button6"] = True

        if st.session_state["button6"]:
            st.session_state["wei_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button6"] = False
            
        st.write(st.session_state["wei_dir"])

    st.markdown("""
                ## Parametros para Entrenar
                Aqui se modifican los parametros para el entrenamiento del modelo.
                """
                )
    tt_split = st.slider('Tamaño del dataset de entrenamiento', 0.0, 1.0, 0.7)

    models = st.multiselect("Elige los modelos", ["RNN","TCN","NBEATS","TFT"], ["RNN"])

    epoch = st.number_input('Numero de epocas',value=100)

    poss_err = 5
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de la base de datos")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de los resultados") 

    if not st.session_state["wei_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de los pesos") 

    if models:
        poss_err -=1
    else:
        st.error("Por favor seleccione al menos un modelo") 
    
    if epoch > 0:
        poss_err -=1
    else:
        st.error("El numero minimo de epocas es de 1")


    st.markdown("""
                ## Entrenamiento
                Una vez solucionado todos los errores para ejecutar el algoritmo genetico
                de al boton.
                """
                )

    if st.button("Correr Entrenamiento"):
            st.session_state["button7"] = True
    if poss_err == 0 and  st.session_state["button7"]:
        data_names = os.listdir(st.session_state["data_dir"])
        for names in data_names:
            names = names.split('.')[0]
            st.write("#### Base de dato:", data_names[0])
            train,test = data_preprocess(names,st.session_state["data_dir"],tt_split)
            for model_name in models:
                best = pd.read_csv(st.session_state["log_dir"]+"/Results_"+names+"_"+model_name+".csv")
                param, recursive = dict_to_param(best)
                model = Model_selection(model_name, param, recursive, epoch)
                my_bar = st.progress(0, text='Entrenando:Epoca 0')
                for e in range(epoch):
                    model.fit(train, verbose = True,epochs=1)
                    progress = 1*(e+1)/epoch
                    my_bar.progress(progress, text='Entrenando:Epoca '+str(e+1))
                model.save(st.session_state["wei_dir"]+"/Model_"+names+"_"+model_name+".pkl")
                forecast = model.historical_forecasts(test,start=test.get_timestamp_at_point(param[0]),forecast_horizon=param[1],retrain=False,verbose=False)
                st.write(r2_score(test, forecast))
        st.session_state["button7"] = False
    else:
        st.error("hay errores pendientes por solucionar")

def test():
    st.write("# Predecir con el mejor modelo entrenado 	:test_tube:")

    st.markdown(
        """
        Este Modulo es para poder probar el modelo con los mejores parametros entrendo, el cual se obtiene en la pestaña de Entrenamiento.

        ## Carpeta de las Bases de datos, resultados y pesos

        Aqui seleccione las carpetas donde estan los archivos de bases de datos, la carpeta de resultados donde se guardaron 
        los resultados de la pestaña de genetico, la carpeta de pesos donde se guardaron en la pestaña de entrenamiento y
        la pred donde se guardaran las predicciones realizadas.

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
        if st.button("Base de datos"):
                st.session_state["button8"] = True
            
        if st.session_state["button8"]:
            st.session_state["data_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button8"] = False
            
        st.write(st.session_state["data_dir"])
    
    with col2:
        if st.button("Resultados"):
            st.session_state["button9"] = True

        if st.session_state["button9"]:
            st.session_state["log_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button9"] = False
            
        st.write(st.session_state["log_dir"])
    
    with col3:
        if st.button("Pesos"):
            st.session_state["button10"] = True

        if st.session_state["button10"]:
            st.session_state["wei_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button10"] = False
            
        st.write(st.session_state["wei_dir"])

    with col4:
        if st.button("Pred"):
            st.session_state["button12"] = True

        if st.session_state["button12"]:
            st.session_state["pred_dir"] = filedialog.askdirectory(master=root)
            st.session_state["button12"] = False
            
        st.write(st.session_state["pred_dir"])
    st.markdown("""
                ## Parametros para Predecir
                Aqui se modifican los parametros para predecir valores futuros.
                """
                )
    models = st.multiselect("Elige los modelos", ["RNN","TCN","NBEATS","TFT"], ["RNN"])
    tt_split = st.slider('Tamaño del dataset de entrenamiento', 0.0, 1.0, 0.7)
    n_pred = st.number_input('Numero de valores a predecir',value=2)

    if st.button("Predecir valores"):
            st.session_state["button11"] = True

    poss_err = 6
    if not st.session_state["data_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de la base de datos")
    
    if not st.session_state["log_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de los resultados") 

    if not st.session_state["wei_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de los pesos") 

    if models:
        poss_err -=1
    else:
        st.error("Por favor seleccione al menos un modelo") 

    if not st.session_state["pred_dir"] == " ":
        poss_err -=1
    else:
        st.error("Por favor seleccione la carpeta de la pred")
    
    if n_pred > 0:
        poss_err -=1
    else:
        st.error("El numero minimo de predicciones es de 1")

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
    "Inicio": intro,
    "Genetico": genetic,
    "Entrenamiento":train,
    "Predecir":test,
}
st.set_page_config(page_title="Nombre del codigo", page_icon="📈")
demo_name = st.sidebar.selectbox("Elige una función", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()