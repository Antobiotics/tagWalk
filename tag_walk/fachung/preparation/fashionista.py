from glob import glob

import fachung.logger as l
import fachung.configuration as conf


class Fashionista():
    def __init__(self, build=True, readable_labels=False):
        self.df = None
        self.labels = None

        if build:
            self.labels = self.build_labels()
            self.df = self.build()

    @property
    def output_dir(self):
        return (
            conf.BASE_DATA +
            conf.get_config().get(conf.MODE, 'outputs')
        )

    @property
    def data_path(self):
        return (
            conf.BASE_DATA +
            conf.get_config()
            .get(conf.MODE, 'fashionista')
        )

    def get_h5_file_paths(self):
        dir_path = self.data_path + 'compiled'
        return glob(dir_path + "*.h5")

    def build(self):
        pass

    def build_labels(self):
        pass

    def save_labels(self):
        pass

    def save_images(self, dirname='fashionista_images/'):
        path = '/'.join([self.output_dir, dirname])
        l.INFO('Saving images to: %s' %(path))

    def prepare(self, df=True, labels=True, images=True):
        if self.df is None:
            self.df = self.build()
        if self.labels is None:
            self.labels = self.build_labels()

        if df:
            path = '/'.join([self.output_dir, 'fashionista.csv'])
            l.INFO('Saving to: %s' %(path))
            self.df.to_csv(path)

        if labels:
            self.save_labels()

        if images:
            self.save_images()
