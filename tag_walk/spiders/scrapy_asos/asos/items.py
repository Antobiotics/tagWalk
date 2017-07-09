# -*- coding: utf-8 -*-

from scrapy import Item, Field

class AsosBrand(Item):

    name = Field()
    id = Field()
    url = Field()

class AsosItem(Item):

    iid = Field()
    url = Field()

    name = Field()
    perm_name = Field()

    cid = Field()
    brand_id = Field()

    color = Field()

    price_orig = Field()
    price_cur = Field()

    image_url_small = Field()
    image_url_large = Field()

class AsosItemDetails(Item):

    url = Field()

    images = Field()
    cat1 = Field()
    cat2 = Field()

    tags = Field()
