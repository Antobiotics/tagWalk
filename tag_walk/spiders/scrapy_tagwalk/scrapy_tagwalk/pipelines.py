# -*- coding: utf-8 -*-

from sqlalchemy.orm import sessionmaker

from scrapy_tagwalk.items import TagwalkItem
from scrapy_tagwalk.models import Item
from scrapy_tagwalk.models import db_connect, create_clothing_table

class TagwalkPipeline(object):
    def __init__(self):
        engine = db_connect()
        create_clothing_table(engine)
        self.Session = sessionmaker(bind=engine)

    @property
    def item_cls(self):
        raise RuntimeError("item_cls must be set.")

    @property
    def model_cls(self):
        raise RuntimeError("model_cls must be set.")

    @property
    def item_switch(self):
        return {
            TagwalkItem: Item,
        }

    def get_model(self, item):
        for item_cls in self.item_switch:
            if isinstance(item, item_cls):
                return self.item_switch[item_cls]
        raise RuntimeError("Unknown item class")

    def process_item(self, item, spider):
        session = self.Session()
        cls_ = self.get_model(item)
        item_mp = cls_(**item)

        try:
            session.add(item_mp)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item
