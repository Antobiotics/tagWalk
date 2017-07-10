#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scrapy import Request
from scrapy import Selector

from scrapy.spider import CrawlSpider

class TagwalkBaseSpider(CrawlSpider):
    name = "asos"
    allowed_domains = ["asos.com"]
    start_urls = [
        "http://www.asos.com/women/a-to-z-of-brands/cat/?cid=1340"
    ]

    custom_settings = {
        'ITEM_PIPELINES': {
            'scrapy_tagwalk.pipelines.TagwalkPipeline': 1
        }
    }

    def parse(self, response):
        pass
