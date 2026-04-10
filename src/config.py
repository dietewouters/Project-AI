# All hyperparameters and constants
import os

config = {
            # Model Parameters
            "input_dim" : 2049,
            "hidden_dims" : [1024, 512, 256, 128],
            "dropout" : 0,

            # Training Parameters
            "lr" : 0.00001,
            "batch_size" : 64,
            "epochs" : 30,
            "optimizer" : "adam",
            "weight_decay" : 1e-5,
            "patience" : 10,
            "num_workers" : 0,

            # Data Paths
            "data_path" : os.path.join(os.getcwd(), 'data', 'groupadditivity_h298')
        }

config_NN = {
# Model Parameters
            "input_dim" : 2050, # Fragnowski et al. (2021) - "Hybrid Machine Learning Models for Chemical Property Prediction".
            "hidden_dims" : [1024, 512, 256, 128], # Smith et al. (2017) - "ANI-1: An extensible neural network potential".
            "dropout" : 0.1, # Wu et al. (2018) - "MoleculeNet: A Benchmark for Deep Learning Molecular Models"
            "activation" : "silu", # Ramachandran et al. (2017) - "Searching for Activation Functions" & Unke et al. (2021) - "PhysNet: A Neural Network for Predicting Energies".
            "use_layer_norm" : True, #Gorishniy et al. (2021) - "Revisiting Deep Learning Models for Tabular Data".

            # Training Parameters
            "lr" : 0.001,
            "batch_size" : 64,
            "epochs" : 20,
            "optimizer" : "adam",
            "weight_decay" : 1e-5,
            "patience" : 10,
            "num_workers" : 0,

            # Data Paths
            "data_path" : os.path.join(os.getcwd(), 'data', 'groupadditivity_h298')
}

config_VAE = {
    # Model Parameters
    "fp_input_dim": 2048,              # fingerprint
    "hidden_dims": [1024, 512, 256],
    "latent_dim": 32,                 # default 
    "beta": 0.1    ,                  # KL weight

    # Training Parameters
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "optimizer": "adam",
    "num_workers": 0,

    # Data
    "data_path": os.path.join(os.getcwd(), 'data', 'groupadditivity_h298')
}