from tqdm import tqdm
from glob import glob
import pandas as pd

def load_data(path):
    """Load all csvs file from path and concat DataFrame

    Args:
        path (string): Path to folder

    Returns:
        pd.DataFrame: contains all of the data from folder
    """
    files = glob(path + '/*')
    print('Loading data .. ')
    dfs = []
    for file in tqdm(files):
        try:
            dfs.append(pd.read_csv(file, compression='gzip'))
        except pd.errors.EmptyDataError as e:
            continue
    df = pd.concat(dfs, ignore_index=True)
    return df