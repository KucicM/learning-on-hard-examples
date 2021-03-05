from data_engine.sql.models import SQLiteQuery
from config import DATABASE_HOST
import sqlite3


def test_no_arguments():
    connection = sqlite3.connect(DATABASE_HOST)
    table_name = "test_no_arguments"

    query = SQLiteQuery(f"CREATE TABLE {table_name} (id INTEGER);")
    query.exec(connection)

    expected = [("test_no_arguments",)]
    check_query = "SELECT name FROM sqlite_master WHERE name = ?;"
    actual = connection.execute(check_query, (table_name,)).fetchall()

    assert expected == actual


def test_no_arguments_select():
    connection = sqlite3.connect(DATABASE_HOST)
    table_name = "test_no_arguments_select"

    connection.execute(f"CREATE TABLE {table_name} (id INTEGER);")
    connection.execute(f"INSERT INTO {table_name} VALUES (2);")

    expected = [(2,)]
    select = SQLiteQuery(f"SELECT id FROM {table_name};")
    actual = select.fetchall(connection)
    assert expected == actual


def test_arguments():
    connection = sqlite3.connect(DATABASE_HOST)
    table_name = "test_arguments"
    connection.execute(f"CREATE TABLE {table_name} (id INTEGER);")

    expected = 2

    insert = SQLiteQuery(f"INSERT INTO {table_name} (id) VALUES (?);")
    insert.exec(connection, [(expected,)])

    expected = [(expected,)]
    check_query = f"SELECT id FROM {table_name};"
    actual = connection.execute(check_query).fetchall()

    assert expected == actual


def test_multiple_arguments_one_column():
    connection = sqlite3.connect(DATABASE_HOST)
    table_name = "test_multiple_arguments_one_column"
    connection.execute(f"CREATE TABLE {table_name} (id INTEGER);")

    num_values = 10
    insert = SQLiteQuery(f"INSERT INTO {table_name} (id) VALUES (?);")
    insert_values = [(i,) for i in range(num_values)]
    insert.exec(connection, insert_values)

    expected = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
    check_query = f"SELECT id FROM {table_name};"
    actual = connection.execute(check_query).fetchall()
    assert expected == actual


def test_multiple_arguments_multiple_columns():
    connection = sqlite3.connect(DATABASE_HOST)
    table_name = "test_multiple_arguments_multiple_columns"
    connection.execute(f"CREATE TABLE {table_name} (id INTEGER, id2 INTEGER);")

    num_values = 10
    insert = SQLiteQuery(f"INSERT INTO {table_name} (id, id2) VALUES (?, ?);")
    insert_values = [(i, num_values-i) for i in range(num_values)]
    insert.exec(connection, insert_values)

    expected = [(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
    check_query = f"SELECT id, id2 FROM {table_name};"
    actual = connection.execute(check_query).fetchall()
    assert expected == actual
