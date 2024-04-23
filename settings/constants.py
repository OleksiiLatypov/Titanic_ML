from pathlib import Path

root_project_dir = Path(__file__).resolve().parent.parent
print(root_project_dir)
TRAIN_DATA = root_project_dir.joinpath('data').joinpath('train.csv')
MODELS_DIRECTORY = root_project_dir.joinpath('models')
SAVED_ESTIMATOR = [file for file in MODELS_DIRECTORY.iterdir()][0]

