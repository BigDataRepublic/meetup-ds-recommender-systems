import urllib
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
import os

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_ROOT_DIR = '/data'
DATA_DIR = os.path.join(DATA_ROOT_DIR, 'ml-100k')
BASE_FILE = os.path.join(DATA_DIR, 'u1.base')
TEST_FILE = os.path.join(DATA_DIR, 'u1.test')
ITEM_FILE = os.path.join(DATA_DIR, 'u.item')


def download_data(url=DATA_URL, data_dir=DATA_ROOT_DIR):
    print("Downloading data set")
    h, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(h, 'r')
    zip_file_object.extractall(data_dir)


def load_data(filename):
    return pd.read_table(filename, names=['user', 'item', 'rating'],
                         dtype={'user': int, 'item': int, 'rating': int},
                         usecols=[0, 1, 2])


class Dataset(object):

    def __init__(self):

        if not (os.path.isfile(BASE_FILE)):
            download_data()

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.ratings = pd.DataFrame()
        self.item_desc = pd.DataFrame()
        self._prepare_ratings()
        self._prepare_item_descriptions()

    def _encode_matrix(self, df):

        df['user'] = self.user_encoder.fit_transform(df['user'])
        df['item'] = self.item_encoder.fit_transform(df['item'])

        self.n_users = df['user'].nunique()
        self.n_items = df['item'].nunique()

        return df

    def get_ratings(self, unary=True, drop_negatives=True, test_data=True, encode=True):
        df = self.ratings.copy()
        if unary:
            df['confidence'] = df['rating']
            df['rating'] = 1. * (df['rating'] >= 3)
            if drop_negatives:
                positives = df['rating'] > 0
                df.drop(index=df[~positives].index.values, inplace=True)

        if not test_data:
            test = df['set'] == 'test'
            df.drop(index=df[test].index.values, inplace=True)
            df.drop(columns=['set'], inplace=True)

        if encode:
            df = self._encode_matrix(df)

        return df

    def get_descriptions(self):
        return self.item_desc.copy()

    def _prepare_ratings(self):

        train = load_data(BASE_FILE)
        train['set'] = 'train'

        test = load_data(TEST_FILE)
        test['set'] = 'test'

        df = pd.concat([train, test], axis=0)

        df = self._encode_matrix(df)

        df['rating'] = df['rating'].astype(float)

        self.ratings = df

    def _prepare_item_descriptions(self):

        item_desc = pd.read_table(ITEM_FILE,
                                  header=None, names=['item', 'title'],
                                  dtype={'item': int, 'title': str}, sep='|',
                                  usecols=[0, 1], encoding='windows-1252')

        item_desc['item'] = self.item_encoder.transform(item_desc['item'])
        item_desc.set_index('item', inplace=True)

        self.item_desc = item_desc['title']


class Recommender(BaseEstimator):

    def __init__(self, dims):
        self.n_users, self.n_items = dims
        self.reconstruction_err_ = np.nan
        self.metrics = {}

    def fit(self, user_item, ratings):
        raise NotImplementedError

    def predict(self, user_item):
        raise NotImplementedError

    def score(self, user_item):
        return self.average_percentile_rank(user_item)

    def average_percentile_rank(self, user_item):
        return self.percentile_rank(user_item).mean()

    def percentile_rank(self, user_item):
        raise NotImplementedError

    def mean_squared_error(self, user_item, ratings):
        predictions = self.predict(user_item)
        return ((ratings - predictions) ** 2).mean()

    def evaluate(self, user_item, ratings):
        self.metrics['re'] = self.reconstruction_err_
        print("reconstruction error: {:.2f}".format(self.metrics['re']))

        self.metrics['mse'] = self.mean_squared_error(user_item, ratings)
        print("mean squared error: {:.4f}".format(self.metrics['mse']))

        self.metrics['apr'] = self.average_percentile_rank(user_item)
        print("average percentile rank: {:.2f}".format(self.metrics['apr']))

        return self.metrics
