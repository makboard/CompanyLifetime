import pickle
import os

def save_pickle(path: str,
                file_name: str,
                data: any) -> None:
    '''Save data as pickle file'''
    with open(os.path.join(path, file_name), 'wb') as fp:
        pickle.dump(data, fp)
        
def open_pickle(path: str,
                file_name: str) -> any:
    '''Load data from pickle file'''
    with open(os.path.join(path, file_name), 'rb') as fp:
        df = pickle.load(fp)
    return df