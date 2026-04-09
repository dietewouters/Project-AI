# All hyperparameters and constants
import os

config = {
            # Model Parameters
            "input_dim" : 1025,
            "hidden_dims" : [512, 256, 128, 64],
            "dropout" : 0.2,

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