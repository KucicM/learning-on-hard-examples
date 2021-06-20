from enum import Enum


class Query(str, Enum):
    CREATE_MAIN_TABLE = """
        CREATE TABLE idx_loss (
            idx INTEGER PRIMARY KEY,
            loss REAL NULL
        );"""

    INSERT_OR_REPLACE_IDX_LOSS = "INSERT OR REPLACE INTO idx_loss (idx, loss) VALUES (?, ?);"
    SELECTIVE_BACKPROP = "SELECT idx FROM idx_loss ORDER BY loss * rand() DESC " + \
                         "LIMIT (SELECT CAST(ROUND(COUNT(*) * ?) AS INTEGER) FROM idx_loss);"

    ROBUST_SELECTIVE_BACKPROP = "WITH tmp AS (SELECT AVG(loss) + (STD(loss) * ?) AS threshold FROM idx_loss) " + \
                                "SELECT idx FROM idx_loss, tmp WHERE loss <= threshold ORDER BY loss * rand() DESC " + \
                                "LIMIT (SELECT CAST(ROUND(COUNT(*) * ?) AS INTEGER) FROM idx_loss);"

