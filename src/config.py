# All hyperparameters and constants
import os

config = {
            # Model Parameters
            "input_dim" : 2049,
            "hidden_dims" : [1024, 512, 256],
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