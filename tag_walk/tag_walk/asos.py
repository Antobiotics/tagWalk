import re
#import pandas as pd

import tag_walk.logger as l

from tag_walk.postgres import AsosConnection

IID_CLEAN_REGEEX = re.compile(r'(.*)&.*$')

def clean_brand_id(brand_id):
    return brand_id.replace('&via=top', '')

def clean_iid(iid):
    res = re.search(IID_CLEAN_REGEEX, iid)
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

    def get_brands(self):
        df = self.get_table_as_pandas('public', 'brand')
        df['id'] = df['id'].apply(clean_brand_id)
        return df.drop_duplicates()

    def get_items(self):
        df = self.get_table_as_pandas('public', 'item')
        df['brand_id'] = df['brand_id'].apply(clean_iid)

        df['price_orig'] = df['price_orig'].apply(clean_brackets)
        df['price_cur'] = df['price_cur'].apply(clean_brackets)

        return df.drop_duplicates()

    def get_item_details(self):
        df = self.get_table_as_pandas('public', 'item_details')

        df['cat1'] = df['cat1'].apply(clean_item_category)

        return df

    def prepare(self):
        print self.get_brands()
        l.INFO("Preparing Asos data")
