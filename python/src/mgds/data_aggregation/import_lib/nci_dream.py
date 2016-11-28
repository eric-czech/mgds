
import os
import pandas as pd

CACHE_DIR = '/Users/eczech/.synapseCache'


def get_all_dream_files():
    res = []
    for root, dirs, files in os.walk(CACHE_DIR):
        for file in files:
            if file.endswith(".txt"):
                res.append(os.path.join(root, file))
    return res


def get_file(search_string):
    for file in get_all_dream_files():
        if search_string in file:
            return file
    raise ValueError('Search string "{}" not found in local NIC DREAM file names'.format(search_string))


def convert_hgnc_id(hgnc_ids):
    """ Converts a handful of known, old symbol names to their more recent counterparts.

    See the nci_dream/gene-copy-number notebook for more details on how these new labels were determined

    :param hgnc_ids: Series of HGNC ID values (should be in original case)
    :return: Converted Series of ids
    """
    m_gene = {
        'Rgr': 'RGL4',
        'Rg9mtd1': 'FAM172BP',
        'RG9MTD1': 'TRMT10C'
    }

    def convert(v):
        if pd.isnull(v):
            return None
        return m_gene.get(v, v)

    return hgnc_ids.apply(convert)
