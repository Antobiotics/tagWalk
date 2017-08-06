import abc

import pandas as pd

import fachung.logger as l
from fachung import configuration


try:
    import psycopg2
    import psycopg2.errorcodes
except ImportError:
    l.WARN("Loading postgres module without psycopg2 installed. "
           "Will crash at runtime if postgres functionality is used.")


def result_iterator(cursor, arraysize=1000):
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        for result in results:
            yield result


class PostgreSQLConnection(object):
    def __init__(self):
        try:
            connection = psycopg2.connect(
                host=self.get_host().split(':')[0],
                port=self.get_port(),
                database=self.get_database(),
                user=self.get_user(),
                password=self.get_password())
            connection.set_client_encoding('utf-8')
        except psycopg2.Error as e:
            l.ERROR(e)

        self.connection = connection

    @abc.abstractproperty
    def instance(self):
        raise RuntimeError("Missig property instance")

    def get_config(self, key):
        return (
            configuration
            .get_config()
            .get(self.instance, key)
        )

    def get_host(self):
        return self.get_config('host')

    def get_user(self):
        return self.get_config('user')

    def get_password(self):
        return self.get_config('password')

    def get_database(self):
        return self.get_config('database')

    def get_port(self):
        return self.get_config('port')

    def fetchall(self, query):
        l.INFO(query)
        cursor = self.connection.cursor()
        records = None
        try:
            cursor.execute(query)
            records = cursor.fetchall()
            cursor.close()
            self.connection.commit()
        except psycopg2.DatabaseError as e:
            l.ERROR(e)
            raise RuntimeError(e)
        return records

    def get_iterator(self, query):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
        except psycopg2.DatabaseError as e:
            l.ERROR(e)
        return result_iterator(cursor)

    def execute(self, command):
        l.INFO('Executing Command')
        l.INFO(command)
        cursor = self.connection.cursor()
        try:
            cursor.execute(command)
            cursor.close()
            self.connection.commit()
        except psycopg2.DatabaseError as e:
            l.ERROR(e)
            raise RuntimeError(e)

    def get_table_definition(self, schema, table):
        command = """
                  SET SEARCH_PATH TO '$user', public, {schema};
                  SELECT column_name, data_type
                  FROM information_schema.columns
                  WHERE table_schema = '{schema}'
                  AND table_name   = '{table}'
                  """.format(schema=schema,
                             table=table)
        return self.fetchall(command)

    def get_table_columns(self, schema, table):
        table_def = self.get_table_definition(schema, table)
        return [c[0] for c in table_def]

    def get_table_as_pandas(self, schema, table):
        columns = self.get_table_columns(schema, table)
        query = """
                SELECT * FROM {schema}.{table}
                """.format(schema=schema, table=table)
        data = self.fetchall(query)
        return pd.DataFrame(data, columns=columns)

class AsosConnection(PostgreSQLConnection):
    @property
    def instance(self):
        return 'asos_db'
