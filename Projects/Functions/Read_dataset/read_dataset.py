import pandas as pd

def read_dataset(dataset_path:str):
    return pd.read_csv(dataset_path)