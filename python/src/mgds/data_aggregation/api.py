
import pandas as pd
import numpy as np
from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src
from mgds.data_aggregation import entity
from mgds.data_aggregation import data_type as dat_typ

import logging
logger = logging.getLogger(__name__)

def get_hugo_gene_ids():
    """ Get HUGO Gene Symbols"""
    d = db.load(src.HUGO_v1, db.IMPORT, 'gene-meta')
    d = d[~d['Approved Name'].str.lower().str.contains('symbol withdrawn|entry withdrawn')]
    d = d['Approved Symbol'].unique()
    return d


def get_entity_mapping(entity_type):
    if entity_type == entity.CELL_LINE:
        return db.load(src.MGDS_v1, db.ENTITY, 'cellline-ids-by-typ')
    if entity_type == entity.PRIMARY_SITE:
        return db.load(src.MGDS_v1, db.ENTITY, 'primary-site-by-src')
    else:
        raise ValueError('Mappings for entity type "{}" not yet supported'.format(entity_type))


def get_cellline_metadata(sources, mappings=None):
    d_id = []
    for source in sources:
        d = get_raw_genomic_data(source, dat_typ.CELLLINE_META, mappings=mappings)
        if d is None:
            continue
        d_id.append(d.assign(SOURCE=source))
    d_id = pd.concat(d_id)
    assert np.all(d_id['CELL_LINE_ID:MGDS'].notnull()), \
        'Found null MGDS ID for cell line in metadata -- '\
        'this should not be possible and reflects a mapping omission or error'
    return d_id


def get_raw_genomic_data(source, data_type, cell_line_taxonomy='COMMON', mappings=None):

    if not db.exists(source, db.IMPORT, data_type):
        return None

    d = db.load(source, db.IMPORT, data_type)

    # Determine the field (based on the cell line taxonomy) that should be joined to in order to get normalized ids
    c_cl_id = 'CELL_LINE_ID{}'.format('' if cell_line_taxonomy == 'COMMON' else ':' + cell_line_taxonomy)

    # If this data contains the necessary id field, join it on the normalized id mappings
    if c_cl_id in d:

        # Load cell line mapping data and subset to source, data type, and taxonomy
        if mappings is not None and entity.CELL_LINE in mappings:
            d_id = mappings[entity.CELL_LINE]
        else:
            d_id = get_entity_mapping(entity.CELL_LINE)

        assert cell_line_taxonomy in d_id, \
            'Cell line taxonomy "{}" not found in mapping data'.format(cell_line_taxonomy)
        d_id = d_id[cell_line_taxonomy]
        assert source in d_id, \
            'Source "{}" not found in cell line mapping data for taxonomy "{}"'.format(src, cell_line_taxonomy)
        d_id = d_id[source]
        assert data_type in d_id, \
            'Data type "{}" not found in cell line mapping data for taxonomy "{}" and source "{}"'\
            .format(data_type, cell_line_taxonomy, source)
        d_id = d_id[data_type].reset_index().dropna()

        # Cell line mapping data now has MGDS_ID and source specific identifiers as columns, so first rename the
        # source specific id field to the same name it should join to in the original data
        d_id = d_id.rename(columns={data_type: c_cl_id, 'MGDS_ID': 'CELL_LINE_ID:MGDS'})

        d = pd.merge(d, d_id, on=c_cl_id, how='left')

    if 'CELL_LINE_ID:MGDS' in d:
        assert 'PRIMARY_SITE:MGDS' not in d, 'Data should not already contain a normalized primary site field'
        assert 'PRIMARY_SITE:SOURCE' not in d, 'Data should not already contain a source specific primary site field'

        # Load primary site mapping data
        if mappings is not None and entity.PRIMARY_SITE in mappings:
            d_ps = mappings[entity.PRIMARY_SITE]
        else:
            d_ps = get_entity_mapping(entity.PRIMARY_SITE)

        # If the source is present in primary site mapping, join this mapping to the data
        if source in d_ps:
            d_ps = d_ps[source].reset_index().rename(columns={source: 'PRIMARY_SITE:SOURCE'})
            # Merge to primary site data, resulting in two extra fields (PRIMARY_SITE:[SOURCE|MGDS])
            d = pd.merge(d, d_ps, on=['CELL_LINE_ID:MGDS'], how='left')

        # Otherwise, log a warning that primary sites will not be available for this source + data type
        else:
            logger.warning(
                'Genomic data for source "{}" and data type "{}" has a cell line mapping '
                'but does not have a primary site mapping (so all primary site fields will be null)'
                .format(source, data_type)
            )
            for c in ['PRIMARY_SITE:SOURCE', 'PRIMARY_SITE:MGDS']:
                d[c] = None

    return d


