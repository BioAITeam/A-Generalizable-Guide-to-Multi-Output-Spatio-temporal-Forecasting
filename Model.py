from darts.models import (
    BlockRNNModel,
    TCNModel,
    TFTModel,
    NBEATSModel,
)

from darts.utils.likelihood_models import QuantileRegression


Global_parameters = [
    (1, 60), # Input lens for all models
    (1, 20), # Output lens for all models

    #RNN
    (0, 2), #RNN unit 
    (2, 100),# Hidden dim
    (1, 5),# N layers (RNN unit)
    (0.1,0.5), # Dropout

    #TCN
    (1, 3), # Dilations
    (2, 5), # Kernel size
    (8, 300), # N filters
    
    #NBEATS
    (2, 6), # N blocks
    (2, 8), # N layers
    (32,1024), # Layer widths
    (1, 80), # Batch size

    #TFT
    (32,256), # Hidden size
    (1, 3), # LSTM layers
    (1, 5) # Attentions Heads
    #This uses Dropout an Batch size from previous 
]

def choose_parameters(Model_name = "RNN"):
  if Model_name not in ["RNN","TCN","NBEATS","TFT"]:
    print("Model name incorrect")
    return[] 
  idx_param = {"RNN" : [0,1,2,3,4,5],
              "TCN" : [0,1,5,6,7,8],
              "NBEATS":[0,1,9,10,11,12],
              "TFT":[0,1,13,14,15,5,12]
  }
  return [Global_parameters[i] for i in idx_param[Model_name]]

def Model_selection(model_name, param,recursive = False, epochs = 20):
  if recursive:
    param[1] = 1
  if model_name == "RNN":
    Model_unit = ["LSTM","GRU","RNN"]
    model = BlockRNNModel(
        input_chunk_length = param[0],
        output_chunk_length = param[1], 
        model= Model_unit[param[2]],
        hidden_dim=param[3],
        n_rnn_layers=param[4],
        #training_length=param[5],
        n_epochs=epochs,
    )
  elif model_name == "TCN":
    model = TCNModel(
        input_chunk_length=param[0],
        output_chunk_length=param[1],
        n_epochs=epochs,
        dropout=param[2],
        dilation_base=param[3],
        weight_norm=True,
        kernel_size=param[4],
        num_filters=param[5],
        random_state=0,
    )
  elif model_name == "NBEATS":
    model = NBEATSModel(
        input_chunk_length=param[0],
        output_chunk_length=param[1],
        generic_architecture=False,
        num_blocks=param[2],
        num_layers=param[3],
        layer_widths=param[4],
        n_epochs=epochs,
        nr_epochs_val_period=1,
        batch_size=param[5],
        model_name="nbeats_interpretable_run",
    )
  elif model_name == "TFT":
    quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 
                 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    model =  TFTModel(
        input_chunk_length=param[0],
        output_chunk_length=param[1],
        hidden_size=param[2],
        lstm_layers=param[3],
        num_attention_heads=param[4],
        dropout=param[5],
        batch_size=param[6],
        n_epochs=epochs,
        add_relative_index=True,
        add_encoders=None,
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
    )
  else:
    print("Model name incorrect")
    model = []
  return model