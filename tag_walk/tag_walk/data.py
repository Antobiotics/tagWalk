import scipy.io

import pandas as pd


PAPERDOLL_DATA_PATH = '../data/paperdoll/data/'

class PaperDoll():
    """
    Simple class to handle the paperdoll dataset

    Usage:
    > paperdoll = PaperDoll()
    > paperdoll_df = paperdoll.df

    """

    def __init__(self, data_path=PAPERDOLL_DATA_PATH):
        self.data_path = data_path
        self.mat = self.load_mat()

        self.labels = self.build_labels()
        self.df = self.build()

    def load_mat(self):
        return scipy.io.loadmat(self.data_path)

    def build(self):
        df = pd.DataFrame(self.mat['samples'][0])

        df['id'] = df['id'].apply(lambda row: row[0][0])
        df['url'] = df['url'].apply(lambda row: row[0])
        df['post_url'] = df['post_url'].apply(lambda row: row[0])
        df['tagging'] = df['tagging'].apply(lambda row: row[0])

        def lookup_tags(row):
            return [self.get_label_name(t) for t in row]

        df['labels'] = df['tagging'].apply(lookup_tags)

        return df

    def build_labels(self):
        df = (
            pd
            .DataFrame(self.mat['labels'])
            .transpose()
        )

        def rowIndex(row):
            return row.name

        df['index'] = df.apply(rowIndex, axis=1)
        df.columns = ['label', 'index']
        df['label'] = df['label'].apply(lambda row: row[0])

        return df

    def get_label_index(self, label_name):
        return self.labels[
            self.labels.label == label_name
        ].index[0]

    def get_label_name(self, label_index):
        return self.labels[
            self.labels.index == label_index
        ].label.values
