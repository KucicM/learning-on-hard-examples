from unittest import TestCase
from sql.models import SQLiteQuery
from config import DATABASE_HOST
import sqlite3


class SQLiteQueryTest(TestCase):
    connection = sqlite3.connect(DATABASE_HOST)

    def test_no_arguments(self):
        table_name = 'test_no_arguments'

        query = SQLiteQuery(f'CREATE TABLE {table_name} (id INTEGER);')
        query.exec(self.connection)

        check_query = 'SELECT name FROM sqlite_master WHERE name = ?;'
        ret = self.connection.execute(check_query, (table_name,)).fetchall()

        self.assertEqual(1, len(ret))

    def test_no_arguments_select(self):
        table_name = 'test_no_arguments_select'

        self.connection.execute(f'CREATE TABLE {table_name} (id INTEGER);')
        self.connection.execute(f'INSERT INTO {table_name} VALUES (2);')

        select = SQLiteQuery(f'SELECT id FROM {table_name};')
        ret = select.fetchall(self.connection)
        self.assertEqual(1, len(ret))
        self.assertEqual(1, len(ret[0]))
        self.assertEqual(2, ret[0][0])

    def test_arguments(self):
        table_name = 'test_arguments'
        self.connection.execute(f'CREATE TABLE {table_name} (id INTEGER);')

        expected = 2

        insert = SQLiteQuery(f'INSERT INTO {table_name} (id) VALUES (?);')
        insert.exec(self.connection, [(expected,)])

        check_query = f'SELECT id FROM {table_name};'
        ret_list = self.connection.execute(check_query).fetchall()

        self.assertEqual(1, len(ret_list))
        ret_tuple = ret_list[0]

        self.assertEqual(1, len(ret_tuple))

        ret = ret_tuple[0]
        self.assertEqual(2, ret)

    def test_multiple_arguments_one_column(self):
        table_name = 'test_multiple_arguments_one_column'
        self.connection.execute(f'CREATE TABLE {table_name} (id INTEGER);')

        num_values = 10
        insert = SQLiteQuery(f'INSERT INTO {table_name} (id) VALUES (?);')
        insert_values = [(i,) for i in range(num_values)]
        insert.exec(self.connection, insert_values)

        check_query = f'SELECT id FROM {table_name};'
        ret_list = self.connection.execute(check_query).fetchall()
        self.assertEqual(num_values, len(ret_list))
        self.assertTrue(all(len(r) == 1 for r in ret_list))
        self.assertTrue(all(r[0] == i for i, r in enumerate(ret_list)))

    def test_multiple_arguments_multiple_columns(self):
        table_name = 'test_multiple_arguments_multiple_columns'
        self.connection.execute(f'CREATE TABLE {table_name} (id INTEGER, id2 INTEGER);')

        num_values = 10
        insert = SQLiteQuery(f'INSERT INTO {table_name} (id, id2) VALUES (?, ?);')
        insert_values = [(i, num_values-i) for i in range(num_values)]
        insert.exec(self.connection, insert_values)

        check_query = f'SELECT id, id2 FROM {table_name};'
        ret_list = self.connection.execute(check_query).fetchall()
        self.assertEqual(num_values, len(ret_list))
        self.assertTrue(all(len(r) == 2 for r in ret_list))
        self.assertTrue(all(r[0] == i and r[1] == num_values-i for i, r in enumerate(ret_list)))
