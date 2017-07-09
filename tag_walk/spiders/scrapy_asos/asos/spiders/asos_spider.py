#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from scrapy import Request
from scrapy import Selector

from scrapy.spider import CrawlSpider

from asos.items import AsosItem, AsosBrand, AsosItemDetails

BRAND_URL_RE = (
    re
    .compile(r'http://www.asos.com/women/a-to-z-of-brands/(.*)/cat/\?cid=(.*)')
)

NEXT_URL_RE = (
    re
    .compile(r'.*(&pge=.*)')
)

ITEM_URL_RE = (
    re
    .compile(r'http://www.asos.com/.*/(.*)/.*/.*?iid=(.*)&clr=(.*)&cid=(.*)')
)


class AsosItemDetailsSpider(CrawlSpider):
    name = "asos_item_details"
    allow_domains = ["asos.com"]

    custom_settings = {
        'ITEM_PIPELINES': {
            'asos.pipelines.ItemDetailPipeline': 3
        }
    }

    def parse(self, response):
        selector = Selector(response)

        images = (
            selector
            .css('.thumbnails')
            .xpath('//img/@src')
            .extract()
        )

        thumbnails = []
        for image in images:
            if '&fit=constrain' in image:
                thumbnails.append(image)

        img_url_re = (
            re
            .compile('(http://images.asos-media.com/.*/.*/.*)(&wid=.*&fit=constrain)')
        )

        for i, t in enumerate(thumbnails):
            thumbnails[i] =  img_url_re.split(t)[1]

        details = AsosItemDetails()
        details['url'] = response.url
        details['images'] = thumbnails

        cat_url = (
            selector
            .css('.product-description')
            .xpath('./span/a[1]/@href')
            .extract()
        )
        print cat_url

        if len(cat_url) != 0:
            print cat_url
            cat_url = cat_url[0]

            cat_url_re = (
                re
                .compile(r'/women/(.*)/(.*)/.*\?cid=.*')
            )

            res = cat_url_re.search(cat_url.lower())
            if res:
                details['cat1'] = res.group(1)
                details['cat2'] = res.group(2)

            tags = (
                selector
                .css('.product-description')
                .xpath('./span/ul/li/text()')
                .extract()
            )

            details['tags'] = tags
        yield details


class AsosBrandSpider(CrawlSpider):
    name = "asos_brand"
    allow_domains = ["asos.com"]

    custom_settings = {
        'ITEM_PIPELINES': {
            'asos.pipelines.ItemPipeline': 2
        }
    }

    def parse(self, response):
        selector = Selector(response)

        product_containers = (
            selector
            .css('.product-container')
        )

        for container in product_containers:
            item = AsosItem()
            res = BRAND_URL_RE.search(response.url)
            if res:
                item['brand_id'] = res.group(2).replace('&via=top', '')


            item['image_url_small'] = (
                container
                .css('.img-wrap')
                .css('.product-img')
                .xpath('@src')
                .extract()
            )[0]

            item['name'] = (
                container
                .css('.name-fade')
                .xpath('./span/text()')
                .extract()
            )[0]

            item['price_cur'] = (
                container
                .css('.scm-pricelist')
                .css('.price-current')
                .xpath('./span[@class="price"]/text()')
                .extract()
            )[0]

            item['price_orig'] = (
                container
                .css('.scm-pricelist')
                .css('.price-previous')
                .xpath('./span[@class="price"]/text()')
                .extract()
            )

            if item['price_orig'] == []:
                item['price_orig'] = ""


            item['url'] = (
                container
                .css('.product-link')
                .css('a::attr(href)')
                .extract()
            )[0]

            res = ITEM_URL_RE.search(item['url'])
            if res:
                item['perm_name'] = res.group(1)
                item['iid'] = res.group(2)
                item['color'] = res.group(3)
                item['cid'] = res.group(4)
            yield item
            #print item
            yield Request(item['url'], callback=AsosItemDetailsSpider().parse)

        next_page = (
            selector
            .css('.pager')
            .css('.next')
            .css('a::attr(href)')
        ).extract()

        if len(next_page) != 0:
            next_page = next_page[0]
            res = NEXT_URL_RE.search(next_page)
            if res:
                base_split = (
                    re.compile('(http://.*)(&pge=.*)')
                    .split(response.url)
                )
                #print base_split
                next_url = base_split[0] + res.group(1)

                if len(base_split) > 1:
                    next_url = base_split[1] + res.group(1)

                #print "Next url: %s"  %(next_url)
                yield Request(next_url, callback=AsosBrandSpider().parse)


class AsosSpider(CrawlSpider):
    name = "asos"
    allowed_domains = ["asos.com"]
    start_urls = [
        "http://www.asos.com/women/a-to-z-of-brands/cat/?cid=1340"
    ]

    custom_settings = {
        'ITEM_PIPELINES': {
            'asos.pipelines.BrandPipeline': 1
        }
    }

    def parse(self, response):

        selector = Selector(response)
        hrefs =  (
            selector
            .xpath('//li//@href')
            .extract()
        )

        brands = []
        for href in hrefs:
            res = BRAND_URL_RE.search(href)
            if res:
                brand = AsosBrand()

                brand['name'] = res.group(1)
                brand['id'] = res.group(2).replace('%via=top', '')
                brand['url'] = href

                brands.append(brand)

        for brand in brands:
            yield brand
            url = brand['url']
            spider = AsosBrandSpider()

            yield Request(url, callback=spider.parse)

