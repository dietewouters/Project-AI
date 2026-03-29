from src.gnn import train_chemprop
train_chemprop("src/cli_workspace/train_prepared.csv", "src/cli_workspace/models/model_h298", target_col="h298", epochs=2, val_csv="src/cli_workspace/val_prepared.csv")
