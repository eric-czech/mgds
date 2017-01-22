
import os
import pandas as pd
from py_utils import io_utils
from mgds.data_aggregation import config
from mgds.data_aggregation import source as src
import logging
logger = logging.getLogger(__name__)

RAW = 'raw'
IMPORT = 'import'
NORMALIZED = 'normalized'
ENTITY = 'entity'
PREP = 'prep'


def get_download_file(source, filename):
    download_dir = os.path.join(config.DATA_DIR, RAW, 'sources', source)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return os.path.join(download_dir, filename)


def raw_file(source, filename):
    return os.path.join(config.DATA_DIR, RAW, '{}_{}'.format(source, filename))


def cache_raw_operation(operation, source, dataset, overwrite=False):
    file_path = _table(source, RAW, dataset)
    return _cache_op(file_path, overwrite, operation)

    # return cache_operation(operation, source, PREP, dataset, overwrite=overwrite)


def cache_prep_operation(operation, dataset, overwrite=False):
    return cache_operation(operation, src.MGDS_v1, PREP, dataset, overwrite=overwrite)


def cache_operation(operation, source, database, dataset, overwrite=False):
    file_path = _table(source, database, dataset)
    return _cache_op(file_path, overwrite, operation)


def _cache_op(file_path, overwrite, operation):
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


def save_obj(obj, source, database, filename):
    file_path = _table(source, database, filename)
    return io_utils.to_pickle(obj, file_path)


def exists(source, database, table):
    file_path = _table(source, database, table)
    return os.path.exists(file_path)


def load(source, database, table):
    file_path = _table(source, database, table)
    return pd.read_pickle(file_path)


def load_obj(source, database, filename):
    file_path = _table(source, database, filename)
    return io_utils.from_pickle(file_path)


def tables(source, database):
    path = os.path.join(config.DATA_DIR, database)
    table_names = []
    for filename in os.listdir(path):
        if not os.path.isfile(os.path.join(path, filename)):
            continue
        if len(filename.split('.')) != 2 and not filename.endswith('.tar.gz'):
            logger.warn(
                'Ignoring invalid table name format for file "{}" '
                '(names should only have one period)'.format(os.path.join(path, filename))
            )
            continue
        if filename.startswith(source + '_'):
            table_names.append(filename.split('.')[0].replace(source + '_', ''))
    return table_names


