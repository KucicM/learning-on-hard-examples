from data_engine.sql.sqlite_query import SQLQuery
from data_engine.sql.sqlite_procedures import init_procedures
from data_engine.utils import unpack_inner_tuple
from typing import List
from config import DATABASE_HOST
import sqlite3
import logging
LOGGER = logging.getLogger(__name__)


class CostRepository:
    """
    Repository for inserting and selecting indices whit largest
    costs, with and without outliers, per class and not.
    """
    __connection: sqlite3.Connection = None

    def __init__(self) -> None:
        LOGGER.info("Creating tables")
        SQLQuery.CREATE_CLASS_COST_TABLE.exec(self.connection)
        SQLQuery.CREATE_TMP_CLASS_COST_TABLE.exec(self.connection)
        LOGGER.info("Creating custom procedures")
        init_procedures(self.connection)

    def insert_indices_classes(self, indices: List[int], classes: List[int]) -> None:
        """
        insert new indices with appropriate class, will throw error if index is already in table
        :param indices: List[int], new indices to insert
        :param classes: List[int], clazz for each index
        :return: None
        """
        indices_clazz = map(tuple, zip(indices, classes))
        SQLQuery.INSERT_INDEX_AND_CLASS.exec(self.connection, indices_clazz)

    def update_costs(self, indices: List[int], costs: List[float]) -> None:
        """
        update each index with new cost
        :param indices: List[int], indices which need to be updated
        :param costs: List[float], cost for each index
        :return: None
        """
        assert len(indices) == len(costs)
        indices_costs = map(tuple, zip(indices, costs))

        SQLQuery.CLEAR_TMP_COST_TABLE.exec(self.connection)
        SQLQuery.INSERT_INDEX_AND_COST_IN_TEMP_TABLE.exec(self.connection, indices_costs)
        SQLQuery.UPDATE_COST_FROM_TEMP_TABLE.exec(self.connection)

    @unpack_inner_tuple
    def select_top_k(self, k: float):
        """
        selects k percent indices which have largest cost.
        :param k: float, cutoff percent, top k percent is selected
        :return: List[int],  list of indices
        """
        assert 0 < k <= 1
        return SQLQuery.SELECT_TOP_K.fetchall(self.connection, k)

    @unpack_inner_tuple
    def select_top_k_per_class(self, k: float):
        """
        selects k percent indices which have largest cost per class.
        :param k: float, cutoff percent, top k percent is selected per class
        :return: List[int], list of indices
        """
        assert 0 < k <= 1
        return SQLQuery.SELECT_TOP_K_PER_CLASS.fetchall(self.connection, k)

    @unpack_inner_tuple
    def select_top_k_without_outliers(self, k: float, number_of_stds: float):
        """
        selects k percent indices which have largest cost but smaller then mean + number_of_stds * std.
        :param k: float, cutoff percent, top k percent is selected
        :param number_of_stds: float, max cost standard deviations from mean that could be selected
        :return: List[int] list of indices
        """
        assert 0 < k <= 1
        assert number_of_stds >= 0
        return SQLQuery.SELECT_TOP_K_WITHOUT_OUTLIERS.fetchall(self.connection, [number_of_stds, k])

    @unpack_inner_tuple
    def select_top_k_per_class_without_outliers(self, k: float, number_of_stds: float):
        """
        selects k percent indices which have largest cost but smaller then mean + number_of_stds * std per class.
        :param k: float, cutoff percent, top k percent is selected per class
        :param number_of_stds: float, max cost standard deviations from mean that could be selected per class
        :return: List[int], list of indices
        """
        assert 0 < k <= 1
        assert number_of_stds >= 0
        return SQLQuery.SELECT_TOP_K_PER_CLASS_WITHOUT_OUTLIERS.fetchall(self.connection, [number_of_stds, k])

    @property
    def connection(self) -> sqlite3.Connection:
        """
        opens connection if is not already opened and returns the connection
        :return: connection
        """
        if self.__connection is not None:
            return self.__connection

        try:
            self.__connection = sqlite3.connect(DATABASE_HOST)
            LOGGER.info(f"Database connection opened on {DATABASE_HOST}")
        except sqlite3.Error as e:
            LOGGER.error("Error while opening connection ", e)

        return self.__connection
