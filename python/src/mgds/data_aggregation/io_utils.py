

import os
import pandas as pd
#import pickle
import dill
from io import StringIO
from zipfile import ZipFile
import urllib

import logging
logger = logging.getLogger(__name__)


def download(url, filepath, check_exists=True):
    """ Download a file if not already downloaded
    :param url: URL for remote file
    :param filepath: Local file path
    :param check_exists: If true, the remote file will only be downloaded if it does not already exist.  If false,
        the local file will be overwritten with newly downloaded contents
    :return: Path to local file containing downloaded data (possibly previously)
    """

    # If the file already exists, return its path only if not overridden via check_exists
    if os.path.exists(filepath):
        if check_exists:
            logger.debug('Returning previously downloaded path for "{}"'.format(filepath))
            return filepath
        else:
            # Otherwise, remove the file since it will be downloaded again
            os.unlink(filepath)

    # If the file does not exist or check_exists was false, download a new version now
    urllib.request.urlretrieve(url, filename=filepath)
    logger.debug('Returning newly downloaded path for "{}"'.format(filepath))

    # Return the path to the downloaded data
    return filepath


def _load_archive_file(archive_path, archive_file):
    with ZipFile(archive_path, 'r') as fd:
        return fd.read(archive_file)


def extract_ftp_zip_to_data_frame(ftp_archive, destination_archive, extract_file, **kwargs):
    r = extract_ftp_zip_to_file_bytes(ftp_archive, destination_archive, extract_file)
    return pd.read_csv(StringIO(r.decode('utf-8')), **kwargs)


def extract_ftp_zip_to_data_frame(ftp_archive, destination_archive, extract_file, **kwargs):
    r = extract_ftp_zip_to_file_bytes(ftp_archive, destination_archive, extract_file)
    return pd.read_csv(StringIO(r.decode('utf-8')), **kwargs)


def extract_ftp_zip_to_file_bytes(ftp_archive, destination_archive, extract_file):
    # If archive has already been downloaded, return bytes for file within archive
    if os.path.exists(destination_archive):
        return _load_archive_file(destination_archive, extract_file)

    # If destination archive does not already exist,
    # download archive from ftp server and write to destination location
    req = urllib.request.Request(ftp_archive)
    with urllib.request.urlopen(req) as response:
        r = response.read()
        with open(destination_archive, 'wb') as fd:
            fd.write(r)

    # Return bytes for desired file within archive
    return _load_archive_file(destination_archive, extract_file)


# Use py_utils.io_utils instead
# def to_pickle(obj, file_path):
#     """ Serialize python object to file
#     :param obj: Object to serialize
#     :param file_path: File path for pickle data
#     :return: File path on success
#     """
#     logger.debug('Writing serialized object to "{}"'.format(file_path))
#     with open(file_path, 'wb') as fd:
#         dill.dump(obj, fd)
#     return file_path
#
#
# def from_pickle(file_path):
#     """ Deserialize python object from file
#     :param file_path: File path for pickle data
#     :return: Deserialized object
#     """
#     logger.debug('Restoring serialized object from "{}"'.format(file_path))
#     with open(file_path, 'rb') as fd:
#         return dill.load(fd)
