import os
import re
import json

from shutil import copyfile

import pandas as pd
from tqdm import tqdm, tqdm_pandas

import fachung.logger as logger
import fachung.configuration as conf


IMAGE_PATH_PREF_RF = r'.*/(images/.*)'

def get_image_path_prefix(local_path):
    match = re.search(IMAGE_PATH_PREF_RF, local_path)
    if match:
        return match.group(1)
    return None


class TagWalk():
    def __init__(self, build=True):
        self.ref_dict = self.init_ref_dict()
        self.ref_dataset = self.build_reference_dataset()

    @property
    def data_dir(self):
        return '/'.join([
            conf.BASE_DATA,
            conf.get_config().get(conf.MODE, 'tag_walk')
        ])

    @property
    def crawl_memory_path(self):
        return '/'.join([
            self.data_dir,
            'crawl_memory.json'
        ])

    @property
    def crawl_memory(self):
        with open(self.crawl_memory_path, 'r') as c_mem:
            return json.load(c_mem)

    @property
    def labels(self):
        return list(self.crawl_memory.keys())

    @property
    def all_images_dir(self):
        return self.data_dir + '/' + 'images/v2/__all/'

    @property
    def df_columns(self):
        return ['designer', 'href', 'name',
                'path', 'season', 'src',
                'label', 'type']

    def init_ref_dict(self):
        ref_dict = {}
        for c in self.df_columns:
            ref_dict[c] = []
        return ref_dict

    def fill_ref_dict(self, image):
        for c in self.df_columns:
            self.ref_dict[c].append(image[c])

    def build_reference_dataset(self):
        logger.INFO("Building reference dataset")

        for label in tqdm(self.labels):
            images = self.crawl_memory[label]['images']
            for image in images:
                image['label'] = label
                image['type'] = 'original'
                self.fill_ref_dict(image)
        ref_df = pd.DataFrame.from_dict(self.ref_dict, orient='columns')

        ref_df['origin_path'] = (
            ref_df['path']
            .apply(get_image_path_prefix)
        )
        ref_df['origin_path'] = self.data_dir + ref_df['origin_path']

        ref_df['destination_path'] = (
            self.all_images_dir +
            ref_df['designer'] + '__' +
            ref_df['season'] + '__' +
            ref_df['name'].apply(lambda x: x.lower().replace(' ' , '_'))
        )

        return ref_df.reset_index(drop=True)

    def build_all_images_dir(self):
        try:
            os.stat(self.all_images_dir)
        except Exception:
            os.mkdir(self.all_images_dir)

    def flatten_images_directory(self):
        tqdm_pandas(tqdm())
        logger.INFO("Flattening images data")
        self.build_all_images_dir()

        self.ref_dataset.progress_apply(
            lambda x: copyfile(x['origin_path'], x['destination_path']),
            axis=1
        )

    def prepare(self, df=True, images=True):
        if df:
            ref_df = self.build_reference_dataset()
            print(ref_df.head())
        if images:
            self.flatten_images_directory()
