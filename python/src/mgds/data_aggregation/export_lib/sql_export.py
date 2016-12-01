
import os
import logging
logger = logging.getLogger(__name__)

EXPORT_PATH = os.path.expanduser('~/data/research/mgds/export/superset')


def export_sql_db(data, table_name, database_name, include_index=False, if_exists='replace'):
    d = data.copy()

    db_path = 'sqlite:///{}/{}.db'.format(EXPORT_PATH, database_name)

    # Replace characters in field names that often cause problems in SQL queries
    d.columns = [c.replace('.', '_').replace(':', '_') for c in d]

    d.to_sql(table_name, db_path, index=include_index, if_exists=if_exists)
    logger.info('Successfully exported data to table "{}" at database path "{}"'.format(table_name, db_path))
    return db_path
