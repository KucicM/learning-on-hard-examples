import cost_repository

if __name__ == '__main__':
    from logging.config import fileConfig
    fileConfig("logging_config.ini", disable_existing_loggers=False)

    repo = cost_repository.CostRepository()
