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


# optimize.py

Implements hyperparameter optimization using  **Optuna** . This module automates the search for the best training configuration by repeatedly training models with different hyperparameters and evaluating their validation performance. In choosing the hyperparameters it keeps the previous trials in mind. This way it makes decision based on which where good before.


Now the amount of trials is only 10 with each 20 epochs. This is too low to actually have good performance, but it would take to long to run on a personal computer.


## create_optuna_objective

Factory function that returns the **objective function** used by Optuna.

### Role

Defines what a single **trial** looks like:

1. Sample a new set of hyperparameters from the search space
2. Build a model with those hyperparameters
3. Train the model
4. Return the validation loss

### Key Details

* A deep copy of `base_config` is used to avoid side effects between trials
* Hyperparameters are sampled using:
  * `suggest_float` for continuous values (e.g. learning rate, dropout)
  * `suggest_categorical` for discrete choices (e.g. batch size, architecture)
* DataLoaders are created inside the objective so `batch_size` can be optimized
* The model is trained using the existing `train()` function
* The objective returns the lowest validation error, not just the one of the last epoch


## run_optimisation

Main entry point for running hyperparameter search.

### Workflow

1. Create an Optuna study with:
   * `direction="minimize"` => objective = lowest validation loss
2. Generate the objective function via `create_optuna_objective`
3. Run multiple trials:

<pre class="overflow-visible! px-0!" data-start="1587" data-end="1669"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>trial → sample hyperparameters → train model → compute validation loss</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

4. Log results
5. Merge best hyperparameters into `base_config`
6. Return `best_config`

## plot_optimisation_results

! Not functional yet !

Visualises the evolution of the optimization process over trials.

#### Purpose

Provides a clear overview of how validation loss changes across trials and whether the optimization is converging.

#### Behaviour

* Plots:
  * validation loss per trial
  * best validation loss found so far (running minimum)
* Only includes completed trials
* Uses Matplotlib for compatibility in any environment (no external UI required)

## plot_param_importance

Estimates and visualises the relative importance of each hyperparameter.

#### Purpose

Identifies which hyperparameters have the strongest impact on model performance.

#### Behaviour

* Uses Optuna’s built-in importance estimation (`get_param_importances`)
* Displays a horizontal bar chart ranking parameters by importance
* Handles cases where importance cannot be computed (e.g. too few trials)

## log_results

Utility function for inspecting optimization results.

### Features

* Prints:
  * best trial number
  * best validation loss
  * best hyperparameters
* Saves all trials to:

<pre class="overflow-visible! px-0!" data-start="2118" data-end="2147"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>optuna_trials.csv</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This allows offline analysis of the search process.

### Output

Returns a dictionary identical to `config`, but with optimized values replacing:

* learning rate
* dropout
* batch size
* hidden layer dimensions

## Limitations

* No pruning is used — all trials run to completion even if clearly suboptimal
* Search space is manually defined and may not cover all optimal regions
* Optimization is based solely on validation loss — no secondary metrics considered
* `weight_decay` is present in config but only effective if used in the optimizer inside `train.py`
* Now only implemented for few trials and epochs => takes long to run

## Future improvements

* Add Optuna pruning for faster optimization
* Expand search space (e.g. optimizer type, weight decay)
* Introduce learning rate schedulers
* Support multi-objective optimization (e.g. accuracy vs. training time)

## Minor Issue

`best_model_state = model.state_dict().copy()` — `state_dict()` returns a dict of tensors that are references, not copies. Use `copy.deepcopy(model.state_dict())` instead to ensure the saved state is not accidentally modified by subsequent training steps.
