
import os
import pandas as pd
from mgds.data_aggregation import io_utils
from mgds.data_aggregation import config

RAW = 'raw'
IMPORT = 'import'
NORMALIZED = 'normalized'
ENTITY = 'entity'


def get_download_file(source, filename):
    download_dir = os.path.join(config.DATA_DIR, RAW, 'sources', source)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return os.path.join(download_dir, filename)


def raw_file(source, filename):
    return os.path.join(config.DATA_DIR, RAW, '{}_{}'.format(source, filename))


def cache_raw_operation(operation, source, dataset, overwrite=False):
    file_path = raw_file(source, dataset + '.pkl')
    if not os.path.exists(file_path) or overwrite:
        obj = operation()
        io_utils.to_pickle(obj, file_path)
    return io_utils.from_pickle(file_path)


def _table(source, database, table):
    return os.path.join(config.DATA_DIR, database, '{}_{}.pkl'.format(source, table))


def save(data, source, database, table):
    file_path = _table(source, database, table)
    data.to_pickle(file_path)
    return file_path


def exists(source, database, table):
    file_path = _table(source, database, table)
    return os.path.exists(file_path)


def load(source, database, table):
    file_path = _table(source, database, table)
    return pd.read_pickle(file_path)
