
import pandas as pd
import numpy as np
import re
import logging
from mgds.data_aggregation import database as db
from mgds.data_aggregation import config as cfg

logger = logging.getLogger(__name__)

GENE = 'gene'
DRUG = 'drug'
CELL_LINE = 'cell_line'
CANCER_SITE = 'cancer_site'
CANCER_HISTOLOGY = 'cancer_histology'

ENTITY_TYPE_ID = {
    'cell_line': 1,
    'gene': 2,
    'drug': 3,
    'cancer_site': 4,
    'cancer_histology': 5
}


ALPHANUM_REGEX = re.compile('[\W_]+')


def remove_non_alphanum(x, replacement=''):
    return ALPHANUM_REGEX.sub(replacement, x)


def get_raw_entities(sources, data_types, col_regex):
    r = {}
    for source in sources:
        for data_type in data_types:
            if source not in r:
                r[source] = {}
            logger.debug('Processing source "{}", data type "{}"'.format(source, data_type))
            if not db.exists(source, db.IMPORT, data_type):
                logger.info('Data for source "{}" and data type "{}" does not exist'.format(source, data_type))
                continue
            r[source][data_type] = db.load(source, db.IMPORT, data_type)\
                .filter(regex=col_regex)\
                .drop_duplicates()\
                .assign(SOURCE=source, DATA_TYPE=data_type)
    return r


def emtpy_mapping_frame():
    return pd.DataFrame({
        'MGDS_ID': [],    # Informative, normalized ID
        'MGDS_NAME': [],  # Display Name
        'TAXONOMY': [],   # HGNC, COSMIC, TCGA
        'SOURCE': [],     # Data source (eg. NCI60, TCGA, GDSC)
        'VALUE': []
    })


def get_entity_mapping(entity_type):
    if not db.exists(cfg.MGDS_SRC, db.ENTITY, entity_type):
        return None
    d = db.load(cfg.MGDS_SRC, db.ENTITY, entity_type)
    validate_mapping(d)
    return d


def validate_mapping(d):
    assert sorted(d.columns.tolist()) == ['MGDS_ID', 'MGDS_NAME', 'SOURCE', 'TAXONOMY', 'VALUE']
    assert d.groupby(['MGDS_ID', 'SOURCE', 'TAXONOMY']).size().max() == 1
    assert d.groupby(['SOURCE', 'TAXONOMY', 'VALUE']).size().max() == 1


def append_entity_mapping(entity_type, d, expect_exists=False):

    validate_mapping(d)
    d_o = get_entity_mapping(entity_type)

    if d_o is None:
        if expect_exists:
            raise AssertionError('Pre-existing mapping for entity type "{}" not found'.format(entity_type))
        d_o = emtpy_mapping_frame()

    assert np.all(d_o['VALUE'].notnull()), \
        'Existing mapping contains null values .. this should not be possible and represents an '\
        'unexpected state than can only be fixed by manually editing the existing mapping table and overwriting it.'
    assert np.all(d['VALUE'].notnull()), 'Mapping updates cannot contain null values'

    if len(d_o) > 0:
        d_m = pd.merge(
            d.rename(columns={'VALUE': 'NEW_VALUE'}),
            d_o.rename(columns={'VALUE': 'OLD_VALUE'}),
            on=['MGDS_ID', 'TAXONOMY', 'SOURCE'],
            how='outer'
        )
    else:
        d_m = d.copy().rename(columns={'VALUE': 'NEW_VALUE'})
        d_m['OLD_VALUE'] = None

    mask_conflict = d_m[(d_m['OLD_VALUE'].notnull()) & (d_m['NEW_VALUE'] != d_m['OLD_VALUE'])]
    if np.any(mask_conflict):
        msg = 'Update contains conflicting values with existing mappings.  Conflicting records (first 25):\n{}'\
            .format(d_m[mask_conflict].head(25))
        raise AssertionError(msg)

    n = len(d_m)
    n_new = (d_m['OLD_VALUE'].isnull()) & (d_m['NEW_VALUE'].notnull())
    n_old = (d_m['OLD_VALUE'].notnull()) & (d_m['NEW_VALUE'].isnull())
    n_same = (d_m['OLD_VALUE'].notnull()) & (d_m['NEW_VALUE'].notnull())
    assert n_new + n_old + n_same == n, \
        'Field counts are not valid (n_new = {}, n_old = {}, n_same = {}, total = {}, n_expected = {}'\
        .format(n_new, n_old, n_same, n_new + n_old + n_same, n)

    def pct(x):
        return round(100 * x / n, 1)
    logger.info(
        'Successfully updated mappings for entity type "{}".  '
        'Update summary:\n'
        'New: {} ({}%)\n'
        'Old: {} ({}%)\n'
        'No Change: {} ({}%}'
        .format(entity_type, n_new, pct(n_new), n_old, pct(n_old), n_same, pct(n_same))
    )

    logger.info('Saving updated entity mapping for type "{}"'.format(entity_type))
    db.save(d_m, cfg.MGDS_SRC, db.ENTITY, entity_type)


    # Return the entire, updated mapping
    return d_m


