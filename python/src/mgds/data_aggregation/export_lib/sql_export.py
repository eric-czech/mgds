
import os
import logging
logger = logging.getLogger(__name__)

EXPORT_PATH = os.path.expanduser('~/data/research/mgds/export/superset')


def export_sql_table(
        data, table_name, database_name, include_index=False, if_exists='replace',
        export_csv=False, export_pickle=False):
    d = data.copy()

    db_path = 'sqlite:///{}/{}.db'.format(EXPORT_PATH, database_name)

    # Replace characters in field names that often cause problems in SQL queries
    d.columns = [c.replace('.', '_').replace(':', '_') for c in d]

    d.to_sql(table_name, db_path, index=include_index, if_exists=if_exists)

    if export_csv:
        csv_path = os.path.join(EXPORT_PATH, database_name + '.csv.db')
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        csv_path = os.path.join(csv_path, table_name + '.csv')
        logger.info('Saving table copy to csv at "{}"'.format(csv_path))
        d.to_csv(csv_path, index=include_index)

    if export_pickle:
        pkl_path = os.path.join(EXPORT_PATH, database_name + '.pkl.db')
        if not os.path.exists(pkl_path):
            os.mkdir(pkl_path)
        pkl_path = os.path.join(pkl_path, table_name + '.pkl')
        logger.info('Saving table copy to pickle file at "{}"'.format(pkl_path))
        d.to_pickle(pkl_path)

    logger.info('Successfully exported data to table "{}" at database path "{}"'.format(table_name, db_path))
    return db_path
