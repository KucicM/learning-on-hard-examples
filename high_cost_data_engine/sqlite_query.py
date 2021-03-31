from enum import Enum


class SQLQuery(str, Enum):
    CREATE_CLASS_COST_TABLE = """
        CREATE TABLE class_cost (
            idx INTEGER PRIMARY KEY,
            class INTEGER NOT NULL,
            cost REAL NULL
        );"""

    CREATE_TMP_CLASS_COST_TABLE = """
        CREATE TABLE tmp_index_cost (
            idx INTEGER PRIMARY KEY,
            cost REAL NULL
        );"""

    INSERT_INDEX_AND_CLASS = "INSERT INTO class_cost (idx, class) VALUES (?, ?);"
    CLEAR_TMP_COST_TABLE = "DELETE FROM tmp_index_cost;"
    INSERT_INDEX_AND_COST_IN_TEMP_TABLE = "INSERT INTO tmp_index_cost (idx, cost) VALUES (?, ?);"
    UPDATE_COST_FROM_TEMP_TABLE = """
        UPDATE class_cost SET cost = 
            (SELECT cost FROM tmp_index_cost WHERE class_cost.idx = tmp_index_cost.idx)
        WHERE class_cost.idx IN (SELECT idx FROM tmp_index_cost);"""

    SELECT_TOP_K = """
        SELECT idx FROM class_cost ORDER BY cost DESC 
        LIMIT (SELECT CAST(ROUND(COUNT(*) * ?) AS INTEGER) FROM class_cost);"""

    SELECT_TOP_K_PER_CLASS = """
        WITH tmp AS (SELECT COUNT(*) AS count_per_class, class FROM class_cost GROUP BY class)
        SELECT idx
        FROM tmp
        JOIN (SELECT idx, cost, class, RANK() OVER (PARTITION BY class ORDER BY cost DESC) rank FROM class_cost) AS tbl
        ON tbl.class = tmp.class
        WHERE rank <= CAST(ROUND(tmp.count_per_class * ?) AS INTEGER);"""

    SELECT_TOP_K_WITHOUT_OUTLIERS = """
        WITH tmp AS (SELECT AVG(cost) + (STD(cost) * ?) AS criteria FROM class_cost)
        SELECT idx 
        FROM class_cost, tmp 
        WHERE cost <= criteria 
        ORDER BY cost DESC 
        LIMIT (SELECT CAST(ROUND(COUNT(*) * ?) AS INTEGER) FROM class_cost);"""

    SELECT_TOP_K_PER_CLASS_WITHOUT_OUTLIERS = """
        SELECT idx
        FROM (
            WITH tmp AS (SELECT class, AVG(cost) + (STD(cost) * ?) AS criteria FROM class_cost GROUP BY class)
            SELECT 
                idx,
                RANK() OVER (PARTITION BY class_cost.class ORDER BY cost DESC) rank,
                COUNT() OVER (PARTITION BY class_cost.class) number_of_examples_per_class
            FROM tmp
            JOIN class_cost
            ON tmp.class = class_cost.class
            WHERE cost <= criteria
        )
        WHERE rank <= CAST(ROUND(number_of_examples_per_class * ?) AS INTEGER);"""
