import os
from mgds.data_aggregation import config


def file(typ, filename, ext='pkl'):
    path = os.path.join(config.DATA_DIR, 'modeling', typ)
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, filename + '.' + ext)
