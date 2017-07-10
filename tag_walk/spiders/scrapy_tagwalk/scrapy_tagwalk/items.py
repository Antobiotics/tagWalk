# -*- coding: utf-8 -*-

from scrapy import Item, Field


class TagwalkItem(Item):
    iid = Field()
    url = Field()
