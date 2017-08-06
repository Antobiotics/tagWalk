import os
import urllib

import scipy.io
import pandas as pd

import fachung.logger as l
import fachung.configuration as conf


class PaperDoll():
    """
    Simple class to handle the paperdoll dataset

    Usage:
    > paperdoll = PaperDoll()
    > paperdoll_df = paperdoll.df

    """

    def __init__(self, build=True, readable_labels=False):
        self.df = None
        self.labels = None

        self.mat = self.load_mat()
        self.readable_labels = readable_labels

        if build:
            self.labels = self.build_labels()
            self.df = self.build()

    @property
    def paperdoll_mat_path(self):
        return conf.get_config().get(conf.MODE, 'paperdoll')

    @property
    def data_path(self):
        return conf.BASE_DATA + self.paperdoll_mat_path

    @property
    def output_dir(self):
        return (
            conf.BASE_DATA +
            conf.get_config().get(conf.MODE, 'outputs')
        )

    def load_mat(self):
        l.INFO("Loading PaperDoll Mat file")
        return scipy.io.loadmat(self.data_path)

    def build(self):
        l.INFO("Building Paperdoll DataFrame")
        df = pd.DataFrame(self.mat['samples'][0])

        df['id'] = df['id'].apply(lambda row: row[0][0])
        df['url'] = df['url'].apply(lambda row: row[0])
        df['post_url'] = df['post_url'].apply(lambda row: row[0])
        df['tagging'] = df['tagging'].apply(lambda row: row[0])

        if self.readable_labels:
            def lookup_tags(row):
                return [self.get_label_name(t) for t in row]

            df['labels'] = df['tagging'].apply(lookup_tags)

        return df

    def build_labels(self):
        l.INFO("Building PaperDoll Labels")
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

    def save_df(self, filename='paperdoll.csv'):
        path = '/'.join([self.output_dir, filename])
        l.INFO('Saving to %s' % path)
        return self.df.to_csv(path, index=False, encoding='utf-8')

    def save_labels(self, filename='paperdoll_labels.csv'):
        path = '/'.join([self.output_dir, filename])
        l.INFO('Saving to %s' % path)
        return self.labels.to_csv(path, index=False, encoding='utf-8')

    def save_images(self, dirname='paperdoll_images/'):
        path = '/'.join([self.output_dir, dirname])
        l.INFO('Saving to %s' % path)

        misses = {}

        def dl_url(row):
            output_path = (
                path +
                str(row.id) +
                '.jpg'
            )
            try:
                if not os.path.isfile(output_path):
                    l.INFO("Retrieving: %s" % row.url)
                    opener.retrieve(row.url, output_path)
                return False
            except Exception as e:
                l.ERROR("%s --> %s" % (e, output_path))
                return True

        opener = urllib.URLopener()
        for _, row in self.df.iterrows():
            err = dl_url(row)
            misses[row.id] = err

        misses_df = pd.DataFrame({'id': misses.keys(),
                                  'status': misses.values()})
        misses_df.to_csv(path+'__meta.csv', index=False)
        self.image_statuses_df = misses_df


    def prepare(self, df=True, labels=True, images=True):
        if self.df is None:
            self.df = self.build()
        if self.labels is None:
            self.labels = self.build_labels()

        if df:
            self.save_df()
        if labels:
            self.save_labels()
        if images:
            self.save_images()
