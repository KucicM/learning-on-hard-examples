from data_engine.database.query import Query
from data_engine.utils import convert_list_of_tuples_to_list
from data_engine.database import sqlite_random
from data_engine.database import sql_std
import sqlite3
import logging
LOGGER = logging.getLogger(__name__)


class LossRepository:
    __connection = None

    def __init__(self, database_host: str):
        self._database_host = database_host
        self._init_cache()
        self._init_functions()
        self._init_aggregate()

    def _init_cache(self) -> None:
        self._execute(Query.CREATE_MAIN_TABLE)

    def _init_functions(self) -> None:
        self._connection.create_function("rand", 0, sqlite_random.rand)

    def _init_aggregate(self) -> None:
        self._connection.create_aggregate("std", 1, sql_std.StandardDeviation)

    def insert_or_replace(self, idx_loss) -> None:
        self._executemany(Query.INSERT_OR_REPLACE_IDX_LOSS, map(tuple, idx_loss))

    @convert_list_of_tuples_to_list
    def selective_backprop(self, cutoff):
        return self._execute(Query.SELECTIVE_BACKPROP, cutoff).fetchall()

    @convert_list_of_tuples_to_list
    def robust_selective_backporp(self, cutoff, std_num):
        return self._execute(Query.ROBUST_SELECTIVE_BACKPROP, [std_num, cutoff]).fetchall()

    @property
    def _connection(self):
        if self.__connection is None:
            self.__connection = self._open_connection()
        return self.__connection

    def _open_connection(self):
        try:
            LOGGER.info(f"Opening database connection on {self._database_host}")
            return sqlite3.connect(self._database_host)
        except sqlite3.Error as e:
            LOGGER.error("Error while opening connection ", e)
            raise e

    def _executemany(self, query: str, args):
        self._connection.executemany(query, args)

    def _execute(self, query: str, arguments=None):
        if arguments is None:
            return self._connection.execute(query)

        arguments = arguments if type(arguments) == list else [arguments]
        return self._connection.execute(query, arguments)

    def __exit__(self):
        try:
            LOGGER.info("Closing database connection")
            self._connection.close()
        except sqlite3.Error as e:
            LOGGER.error("Error while closing connection ", e)
            raise e
