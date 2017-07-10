from sqlalchemy import create_engine, Column, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

import scrapy_tagwalk.settings as settings
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
    images = Column('images', postgresql.ARRAY(String))
