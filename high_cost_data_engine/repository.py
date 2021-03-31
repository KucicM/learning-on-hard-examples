from high_cost_data_engine.sqlite_query import SQLQuery
from high_cost_data_engine.sqlite_procedures import init_procedures
from high_cost_data_engine import utils
from typing import List, Iterator
from config import DATABASE_HOST
import sqlite3
import logging
LOGGER = logging.getLogger(__name__)


class HighCostRepository:
    def __init__(self) -> None:
        self._open_connection()
        self._create_tables()
        init_procedures(self.connection)

    def _create_tables(self):
        self._execute(SQLQuery.CREATE_CLASS_COST_TABLE)
        self._execute(SQLQuery.CREATE_TMP_CLASS_COST_TABLE)

    def insert_new_indices_and_classes(self, indices: List[int], classes: List[int]) -> None:
        assert len(indices) == len(classes)
        indices_classes = map(tuple, zip(indices, classes))
        self._executemany(SQLQuery.INSERT_INDEX_AND_CLASS, indices_classes)

    def update_costs(self, indices: List[int], costs: List[float]) -> None:
        assert len(indices) == len(costs)
        indices_costs = map(tuple, zip(indices, costs))

        self._execute(SQLQuery.CLEAR_TMP_COST_TABLE)
        self._executemany(SQLQuery.INSERT_INDEX_AND_COST_IN_TEMP_TABLE, indices_costs)
        self._execute(SQLQuery.UPDATE_COST_FROM_TEMP_TABLE)

    @utils.convert_list_of_tuples_to_list
    def get_top_k_percent_indices(self, k: float):
        assert 0 < k <= 1
        return self._fetchall(SQLQuery.SELECT_TOP_K, k)

    @utils.convert_list_of_tuples_to_list
    def get_top_k_percent_indices_per_class(self, k: float):
        assert 0 < k <= 1
        return self._fetchall(SQLQuery.SELECT_TOP_K_PER_CLASS, k)

    @utils.convert_list_of_tuples_to_list
    def get_top_k_percent_indices_without_outliers(self, k: float, number_of_stds: float):
        assert 0 < k <= 1
        assert number_of_stds >= 0
        return self._fetchall(SQLQuery.SELECT_TOP_K_WITHOUT_OUTLIERS, [number_of_stds, k])

    @utils.convert_list_of_tuples_to_list
    def get_top_k_percent_indices_per_class_without_outliers(self, k: float, number_of_stds: float):
        assert 0 < k <= 1
        assert number_of_stds >= 0
        return self._fetchall(SQLQuery.SELECT_TOP_K_PER_CLASS_WITHOUT_OUTLIERS, [number_of_stds, k])

    def _execute(self, query: str, arguments=None):
        if arguments is None:
            return self.connection.execute(query)

        arguments = arguments if type(arguments) == list else [arguments]
        return self.connection.execute(query, arguments)

    def _executemany(self, query: str, arguments: Iterator[tuple]):
        self.connection.executemany(query, arguments)

    def _fetchall(self, query: str, arguments):
        return self._execute(query, arguments).fetchall()

    def _open_connection(self):
        try:
            LOGGER.info(f"Opening database connection on {DATABASE_HOST}")
            self.connection = sqlite3.connect(DATABASE_HOST)
        except sqlite3.Error as e:
            LOGGER.error("Error while opening connection ", e)
            raise e

    def __exit__(self):
        try:
            LOGGER.info("Closing database connection")
            self.connection.close()
        except sqlite3.Error as e:
            LOGGER.error("Error while closing connection ", e)
            raise e
