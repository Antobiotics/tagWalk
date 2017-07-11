import os
import re
import urllib

import pandas as pd

import tag_walk.logger as l
import tag_walk.configuration as conf

from tag_walk.postgres import AsosConnection

IID_CLEAN = re.compile(r'(.*)&.*$')
URL_DETAIL_GET_PROD = re.compile(r'.*/(.*)\?iid=(.*)&.*')
URL_DETAIL_GET_GROUP = re.compile(r'.*/(.*)\?sgid=(.*)&.*')
SGID_EXTRACT = re.compile(r'.*&sgid=(.*)')

def extract_sgid(x):
    res = SGID_EXTRACT.search(x)
    if res:
        return res.group(1)
    return None

def extract_iid_from_url(x):
    res = URL_DETAIL_GET_PROD.search(x)
    if res:
        return res.group(1)
    return None

def extract_sgid_from_url(x):
    res = URL_DETAIL_GET_GROUP.search(x)
    if res:
        return res.group(1)
    return None

def clean_brand_id(brand_id):
    return brand_id.replace('&via=top', '')

def clean_iid(iid):
    res = re.search(IID_CLEAN, iid)
    if res:
        return res.group(1)
    return iid

def clean_brackets(element):
    return (
        element
        .replace('}', '')
        .replace('{', '')
    )

def clean_item_category(cat):
    try:
        return cat.split('/')
    except Exception:
        return cat

class Asos(AsosConnection):
    """
    Wrapper class for ASOS database:
        > Denoise data
        > Save as CSV
        > Merge dataset
        > Image downloader utilities
    """

    def __init__(self, build=True, readable_labels=False):
        super(Asos, self).__init__()
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

    def get_brands(self):
        df = self.get_table_as_pandas('public', 'brand')
        df['id'] = df['id'].apply(clean_brand_id)
        return df.drop_duplicates()

    def get_items(self):
        df = self.get_table_as_pandas('public', 'item')
        df['brand_id'] = df['brand_id'].apply(clean_iid)

        df['price_orig'] = df['price_orig'].apply(clean_brackets)
        df['price_cur'] = df['price_cur'].apply(clean_brackets)

        df['sgid'] = df['iid'].apply(extract_sgid)
        df['iid'] = df['iid'].apply(clean_iid)

        return df.drop_duplicates()

    def get_item_details(self):
        df = self.get_table_as_pandas('public', 'item_details')
        df['iid'] = df['url'].apply(extract_iid_from_url)
        df['sgid'] = df['url'].apply(extract_sgid_from_url)

        df['cat1'] = df['cat1'].apply(clean_item_category)

        return df

    def build(self):
        df_brand = self.get_brands()
        df_items = self.get_items()
        df_details = self.get_item_details()

        # We Will loose groups here. One must update the crawler to
        # fetch groups separately
        df = pd.merge(df_items[df_items['sgid'].isnull()],
                      df_details[df_details['sgid'].isnull()],
                      on='iid')
        df_brand.columns = ['brand_id', 'brand_name', 'brand_url']
        return pd.merge(df, df_brand, on='brand_id')

    def build_labels(self):
        return None

    def save_labels(self):
        return None

    def save_images(self, dirname='asos_images/'):
        # TODO: Generalise for reuse
        path = '/'.join([self.output_dir, dirname])
        l.INFO('Saving images to: %s' %(path))

        misses = {}

        def dl_url(row):
            output_path = (
                path +
                str(row.iid)
            )
            try:
                images_urls = list(set(
                    row['images']
                ))
                for i, url in enumerate(images_urls):
                    url = url.replace("$S$", "$XXL$")
                    img_path = (
                        output_path + '__' +
                        str(i) + '.jpg'
                    )
                    if not os.path.isfile(img_path):
                        opener.retrieve(url, img_path)
                return False
            except Exception as e:
                l.ERROR("%s --> %s" % (e, output_path))
                return True

        opener = urllib.URLopener()
        for _, row in self.df.iterrows():
            err = dl_url(row)
            misses[row.iid] = err

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
            path = '/'.join([self.output_dir, 'asos.csv'])
            l.INFO('Saving to: %s' %(path))
            self.df.to_csv(path)

        if labels:
            self.save_labels()

        if images:
            self.save_images()

