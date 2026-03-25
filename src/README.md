# `config.py`

Single source of truth for all hyperparameters and constants. Every other file imports from here — nothing is hardcoded elsewhere.

## Model Parameters
Defines the MLP architecture: input size, hidden layer structure, and dropout rate.

## Training Parameters
Controls the training loop behaviour: learning rate, batch size, number of epochs, optimizer choice, L2 regularisation via `weight_decay`, and early stopping patience.

## Data Paths
Builds file paths dynamically using `os.path.join` and `os.getcwd()` so the project works on any machine regardless of where it is cloned.

# `data_loader.py`

Handles all data loading and preprocessing. Provides a `Dataset` class and a factory function that returns ready-to-use `DataLoader` objects.

## `MoleculeDataset`
A PyTorch `Dataset` that merges two CSV files by row position — one with fingerprints and clean target values, one with molecule names and noisy values. The fingerprint string is parsed character by character into a `(2048,)` binary tensor. An optional `StandardScaler` can be passed to normalise the noisy input and target values.

## `get_dataloaders`
Currently a **debug version** — it loads a single dataset file and splits it 80/20 into train and validation using `random_split`. The test set and index-based splitting are commented out and not yet active.

## Known Limitations
- Works only on a fraction of the data, not the full dataset
- Uses random splitting instead of the pre-computed index files
- Test set is commented out
- Scaler is fit on the full dataset before splitting, meaning val sees training statistics — the scaler should be fit on the training set only once the proper splits are in place

# `evaluate.py`

Contains the evaluation logic used both during training and for final testing.

## `evaluate`
Runs inference on a given loader with gradients disabled and the model in eval mode (dropout turned off). Returns the weighted average loss across all batches, correctly accounting for the smaller last batch.

## `test`
Thin wrapper around `evaluate` that targets `loader["test"]` specifically and prints the result. Should only be called **once**, after hyperparameter optimization is complete — calling it earlier leaks test information into model selection.

# Fingerprints
Everything has been annotated within the files 

# `model.py`

Defines the MLP architecture as a PyTorch `nn.Module`.

## `MLP`
A fully connected neural network for molecular property regression. The architecture is built dynamically from `hidden_dims`, making it fully configurable for hyperparameter optimization. Each hidden layer is followed by ReLU activation and dropout. The output layer produces a single value with no activation, standard for regression.

Default parameters are pulled from `config.py` so the model can be instantiated with no arguments during normal runs, while still accepting overrides during hyperparameter search.

## Current Limitations
- The architecture is a plain stack of identical blocks — no skip connections, no batch normalisation, no attention. These may be added as the design evolves.
- `print(self.network)` is currently inside `__init__`, which means it prints every time a model is instantiated — including during hyperparameter search where hundreds of models are created. This should be moved outside the class or guarded with a flag.

# `train.py`

Implements the full training procedure, from a single gradient update to a complete training run with early stopping.

## `train_one_epoch`
Runs one full pass over the training set. Sets the model to train mode, iterates over batches, computes the loss, backpropagates, and updates the weights. Returns the weighted average training loss across all batches.

## `train`
Full training loop over all epochs. Calls `train_one_epoch` and `evaluate` at each epoch and logs both losses to a history dict. Implements early stopping based on validation loss and restores the best model state at the end.

Optimizer and criterion are optional arguments — if not provided, they default to Adam and MSELoss respectively. This is useful for hyperparameter search where `optimize.py` may want to pass a custom optimizer.

## Good Practices Already in Place
- Optimizer and criterion are injectable, making the function flexible for hyperparameter search
- Best model state is saved and restored, so the returned model is always the best one seen during training, not the last one
- Early stopping is based on validation loss, not training loss

## Minor Issue
`best_model_state = model.state_dict().copy()` — `state_dict()` returns a dict of tensors that are references, not copies. Use `copy.deepcopy(model.state_dict())` instead to ensure the saved state is not accidentally modified by subsequent training steps.