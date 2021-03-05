from dataclasses import dataclass
from typing import Tuple, List, Any


@dataclass
class SQLiteQuery:
    """
    wrapper for calling database with specify query.
    """
    query: str

    def fetchall(self, connection, arguments: Any = None) -> List[Tuple[Any]]:
        """
        executes the query and fetches all the data from the result.
        :param connection: sqlite3 connection
        :param arguments: zero or more values used in query
        :return: returns list of tuples of values from the query
        """
        if arguments is None:
            return connection.execute(self.query).fetchall()

        arguments = arguments if type(arguments) == list else [arguments]
        return connection.execute(self.query, arguments).fetchall()

    def exec(self, connection, arguments: Any = None) -> None:
        """
        executes the query without returning the data from the result.
        It is possible to provide multiple sets of arguments in list of tuples and
        all values will be executed in transactions.
        :param connection: sqlite3 connection
        :param arguments: zero or more values used in query
        :return: None
        """
        if arguments is None:
            connection.execute(self.query)
            return

        connection.executemany(self.query, arguments)

    def __str__(self) -> str:
        return "Query: {}".format(self.query)
