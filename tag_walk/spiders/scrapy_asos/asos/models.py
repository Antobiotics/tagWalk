from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

import asos.settings as settings

DeclarativeBase = declarative_base()

def db_connect():
    """ performs db connect using db settings from settings.py.
        Returns sqlalchemy engine instance
        """
    return create_engine(URL(**settings.DATABASE))

def create_clothing_table(engine):
    DeclarativeBase.metadata.create_all(engine)

class Item(DeclarativeBase):
    __tablename__ = "item"

    iid = Column('iid', String, primary_key=True)
    url = Column('url', String)

    name = Column('name', String)
    perm_name = Column('perm_name', String)

    cid = Column('cid', String)
    brand_id = Column('brand_id', String)

    color = Column('color', String)

    price_orig = Column('price_orig', String, nullable=True)
    price_cur = Column('price_cur', String)

    image_url_small = Column('image_url_small', String)
    image_url_large = Column('image_url_large', String)

class Brand(DeclarativeBase):
    __tablename__ = 'brand'

    id = Column('id', String, primary_key=True)
    name = Column('name', String)
    url = Column('url', String)


class ItemDetails(DeclarativeBase):
    __tablename__ = 'item_details'

    url = Column('url', String, primary_key=True)
    images = Column('images', postgresql.ARRAY(String))
    cat1 = Column('cat1', String)
    cat2 = Column('cat2', String)
    tags = Column('tags', postgresql.ARRAY(String))
