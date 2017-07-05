from sqlalchemy.orm import sessionmaker
from asos.models import Jackets, db_connect, create_clothing_table


class AsosPipeline(object):
    """Asos pipeline for storing scraped items in the database"""
    def __init__(self):
        engine = db_connect()
        create_clothing_table(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """Save jacket items in the database.

        This method is called for every item pipeline component.

        """
        session = self.Session()
        clothing = Clothing(**item)

        # print 'processing item', item['name'], '\n\n'

        try:
            session.add(clothing)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item
