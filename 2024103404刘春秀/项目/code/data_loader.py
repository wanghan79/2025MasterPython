import pandas as pd

def load_data():
    ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    items = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
    data = pd.merge(ratings, items, on='item_id')
    return data

def train_test_split(data, test_ratio=0.2):
    data = data.sample(frac=1).reset_index(drop=True)
    test_size = int(len(data) * test_ratio)
    return data[:-test_size], data[-test_size:]