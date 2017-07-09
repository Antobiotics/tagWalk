#import pandas as pd

import tag_walk.logger as l

from tag_walk.postgres import AsosConnection


class Asos(AsosConnection):

    def get_brands(self):
        return self.get_table_as_pandas('public', 'brand')

    def get_items(self):
        return self.get_table_as_pandas('public', 'item')

    def get_item_details(self):
        return self.get_table_as_pandas('public', 'item_details')


    def prepare(self):
        print self.get_brands()
        l.INFO("Preparing Asos data")
